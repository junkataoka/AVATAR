#####################################################################################
#                                                                                   #
# All the codes about the model constructing should be kept in the folder ./models/ #
# All the codes about the data process should be kept in the folder ./data/         #
# The file ./opts.py stores the options                                             #
# The file ./trainer.py stores the training and test strategy                       #
# The ./main.py should be simple                                                    #
#                                                                                   #
#####################################################################################
import os
import json
import shutil
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from models.model_construct import construct # for the model construction
from trainer import train # for the training process
from trainer import validate, validate_compute_cen # for the validation/test process
from trainer import kernel_k_means # for K-means clustering and its variants
from trainer import source_select # for source sample selection
from opts import opts # options for the project
from utils.prepare_data import generate_dataloader # prepare the data and dataloader
import time
import gc
from collections import defaultdict
from sklearn.manifold import TSNE

args = opts()

best_prec1 = 0
best_test_prec1 = 0
cond_best_test_prec1 = 0
best_cluster_acc = 0
best_cluster_acc_2 = 0
counter = 0
    
def main():
    global args, best_prec1, best_test_prec1, cond_best_test_prec1, best_cluster_acc, best_cluster_acc_2, counter

    # define model
    model = construct(args)
    model = torch.nn.DataParallel(model.cuda()) # define multiple GPUs

    # define learnable cluster centers
    p_label_tar = Variable(torch.cuda.FloatTensor(args.num_classes).fill_(0))
    p_label_src = Variable(torch.cuda.FloatTensor(args.num_classes).fill_(0))
    # Define loss function/criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    np.random.seed(12)  # may fix test data
    random.seed(12)
    torch.manual_seed(12)
    params = []
    base_params = []
    # apply different learning rates to different layer
    if "vit" in args.arch or "dino" in args.arch:
        dfs_freeze_vit(model)
        for k, v in model.named_parameters():
            if not k.__contains__('pred'):
                base_params += [{'params': v, 'name': "feature"}]
            else:
                params += [{'params': v, 'name': "pred"}]
                
    elif "resnet" in args.arch:
        for k, v in model.named_parameters():
            if "fc" not in k:
                base_params += [{'params': v, 'name': "feature"}]
            else:
                params += [{'params': v, 'name': "pred"}]


    optimizer = torch.optim.SGD(base_params,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

    optimizer_cls = torch.optim.SGD(params,
                                    lr=args.lr * 10,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

    learn_cen = Variable(torch.cuda.FloatTensor(args.num_classes, len(params[3])).fill_(0))
    learn_cen.requires_grad_(False)

    # resume
    epoch = 0
    dict_th = defaultdict(list)
    dict_mu = defaultdict(list)
    dict_sd = defaultdict(list)
    dict_acc = defaultdict(list)
    # make log directory
    print(args.log)
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
        print("create new log directory")
    if not os.path.isdir(os.path.join(args.log, "tsne")):
        os.makedirs(os.path.join(args.log, "tsne"))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()

    # process data and prepare dataloaders
    train_loader_source, train_loader_target, val_loader_target, val_loader_target_t, val_loader_source = generate_dataloader(args)
    train_loader_target.dataset.tgts = list(np.array(torch.LongTensor(train_loader_target.dataset.tgts).fill_(-1))) # avoid using ground truth labels of target

    print('begin training')
    batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
    num_itern_total = args.epochs * batch_number

    new_epoch_flag = False # if new epoch, new_epoch_flag=True
    test_flag = False # if test, test_flag=True
    src_cs = torch.cuda.FloatTensor(len(train_loader_source.dataset.tgts)).fill_(1) # initialize source weights
    tar_cs = torch.cuda.FloatTensor(len(train_loader_target.dataset.tgts)).fill_(1) # initialize source weights

    count_itern_each_epoch = 0
    th = torch.zeros(args.num_classes).cuda()
    for itern in range(epoch * batch_number, num_itern_total):
        # evaluate on the target training and test data
        if (itern == 0) or (count_itern_each_epoch == batch_number):
            prec1, c_s, c_t, c_srctar, source_features, source_targets, \
            target_features, target_targets, pseudo_labels, labels_src, labels_tar = validate_compute_cen(val_loader_target_t, val_loader_source, model, criterion, epoch, args)

            test_acc, acc_for_each_class = validate(val_loader_target, model, criterion, epoch, args)
            test_flag = True

            # K-means clustering or its variants
            cen = c_t
            _, c_t = kernel_k_means(target_features, target_targets, pseudo_labels, train_loader_target, epoch, model, args, best_cluster_acc, change_target=False)

            # Re-initialize learnable cluster centers
            cen = (c_t + c_s) / 2# or c_srctar

            learn_cen.data = cen.data.clone()
            tar_prob_class = F.softmax(pseudo_labels, dim=1)
            tar_prob_class = F.softmax(pseudo_labels, dim=1)
            p_label_tar.data = tar_prob_class.mean(0).clone()
            p_label_src.data = labels_src.mean(0).clone()
            p_label_src.data = labels_src.mean(0).clone()

            if itern != 0:
                count_itern_each_epoch = 0
                epoch += 1

            src_cs = source_select(source_features, source_targets, target_features, pseudo_labels, train_loader_source, epoch, c_t.data.clone(), args)
            tar_cs = source_select(target_features, target_targets, source_features, source_targets, train_loader_target, epoch, c_t.data.clone(), args)

            # tsne_feature = torch.cat([source_features, target_features], axis=0)

            # tsne_true_label =torch.cat([source_targets, target_targets], axis=0).view(-1)
            # tsne_pseudo_label =torch.cat([source_targets, pseudo_labels.max(1)[1].long()], axis=0).view(-1)

            # tsne_embed_1 = TSNE(n_components=2).fit_transform(tsne_feature.cpu().numpy())

            # domain_label = [0 for i in range(source_features.shape[0])] + [1 for i in range(target_features.shape[0])]

            # tsne_df = pd.DataFrame(tsne_embed_1)
            # label_df = pd.DataFrame({"True_label": tsne_true_label.cpu().numpy().tolist(),
            #                          "Pseudo_label": tsne_pseudo_label.cpu().numpy().tolist(),
            #                          "Domain label": domain_label})

            # tsne_df.to_csv(f"{args.log}/tsne/tsne1_epoch{epoch}.csv")
            # label_df.to_csv(f"{args.log}/tsne/label_epoch{epoch}.csv")


            # Create threthold
            m = torch.zeros((target_targets.size(0), args.num_classes)).fill_(0).cuda()
            sd = torch.zeros((target_targets.size(0), args.num_classes)).fill_(0).cuda()
            m.scatter_(dim=1, index=target_targets.unsqueeze(1), src=tar_cs.unsqueeze(1).cuda()) # assigned pseudo labels
            sd.scatter_(dim=1, index=target_targets.unsqueeze(1), src=tar_cs.unsqueeze(1).cuda()) # assigned pseudo labels

            for i in range(args.num_classes):
                mu = m[m[:, i] != 0, i].mean()
                sdv = sd[sd[:, i] != 0, i].std()
                th[i] = mu - sdv
                dict_mu[i].append(mu.cpu().numpy())
                dict_sd[i].append(sdv.cpu().numpy())
                dict_th[i].append(th[i].cpu().numpy())

            batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
            train_loader_target_batch = enumerate(train_loader_target)
            train_loader_source_batch = enumerate(train_loader_source)

            del source_features
            del source_targets
            del target_features
            del target_targets
            del pseudo_labels
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        dict_acc["epoch"].append(epoch)
        dict_acc["test_acc"].append(test_acc)
        for i in range(len(acc_for_each_class)):
            dict_acc[f"test_acc_class_{i+1}"].append(acc_for_each_class[i].numpy())

        df_sd = pd.DataFrame.from_dict(dict_sd)
        df_mu = pd.DataFrame.from_dict(dict_mu)
        df_th = pd.DataFrame.from_dict(dict_th)
        df_acc = pd.DataFrame.from_dict(dict_acc)
        df_mu.to_csv(os.path.join(args.log, "df_mu.csv"), index=None)
        df_sd.to_csv(os.path.join(args.log, "df_sd.csv"), index=None)
        df_th.to_csv(os.path.join(args.log, "df_th.csv"), index=None)
        df_acc.to_csv(os.path.join(args.log, "acc.csv"), index=None)

        if test_flag:
            # record the best prec1 and save checkpoint
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            if test_acc > best_prec1:
                counter = 0
                best_prec1 = test_acc
                cond_best_test_prec1 = 0
                cond_best_test_prec1 = 0

                log.write('\n                                                                                 best val acc till now: %3f' % best_prec1)
            else: counter += 1
            if test_acc > best_test_prec1:
                best_test_prec1 = test_acc
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'learn_cen': learn_cen,
                    'best_prec1': best_prec1,
                    'best_test_prec1': best_test_prec1,
                    'cond_best_test_prec1': cond_best_test_prec1,
                }, args)

            test_flag = False

            test_flag = False
        if counter > args.stop_epoch:
                break

        # train for one iteration
        train_loader_source_batch, train_loader_target_batch = train(
            train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, 
            learn_cen, optimizer, optimizer_cls, itern, epoch, src_cs, tar_cs, args, p_label_src, p_label_tar, th)

        #model = model.cuda()
        count_itern_each_epoch += 1

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n***   best val acc: %3f   ***' % best_prec1)
    log.write('\n***   best test acc: %3f   ***' % best_test_prec1)
    log.write('\n***   cond best test acc: %3f   ***' % cond_best_test_prec1)
    # end time
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()

def count_epoch_on_large_dataset(train_loader_target, train_loader_source, args):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    batch_number_s = len(train_loader_source)
    if batch_number_s > batch_number_t:
        batch_number = batch_number_s
    return batch_number


def save_checkpoint(state, args):
    filename = 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)

def dfs_freeze_vit(model):
    for name1, child in model.named_children():
        for name2, param in child.named_parameters():
            if "fc" in name2 or "norm" in name2 or "patch_embed" in name1:
                param.requires_grad = False


if __name__ == '__main__':
    main()


#
