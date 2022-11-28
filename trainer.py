import time
import torch
import os
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.kernel_kmeans import KernelKMeans
import gc
import pandas as pd

def train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, learn_cen, learn_cen_2, optimizer, optimizer_cls, itern, epoch, src_cs, tar_cs, args, p_label_src, p_label_tar, th):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1_source = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1 # penalty parameter

    weight_tar_cluster = lam
    weight_src_cluster = lam
    weight_tar_cls = lam
    weight_dis = lam

    adjust_learning_rate(optimizer, epoch, args) # adjust learning rate
    adjust_learning_rate(optimizer_cls, epoch, args) # adjust learning rate

    end = time.time()
    # prepare target data
    try:
        (input_target, target_target, tar_index) = train_loader_target_batch.__next__()[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        (input_target, target_target, tar_index) = train_loader_target_batch.__next__()[1]

    try:
        (input_source, target_source, index) = train_loader_source_batch.__next__()[1]
    except StopIteration:
        train_loader_source_batch = enumerate(train_loader_source)
        (input_source, target_source, index) = train_loader_source_batch.__next__()[1]

    target_source = target_source.cuda()
    input_source_var = Variable(input_source)
    target_target = target_target.cuda()
    input_target_var = Variable(input_target)

    loss = 0
    f_t, f_t_2, ca_t = model(input_target_var)

    if args.domain_adv:
        d_t_loss = CondDiscriminatorLoss(epoch, ca_t, tar_index, tar_cs, src=False, dis_cls=False)
        loss += weight_dis * d_t_loss

    if args.dis_tar:
        tardis_loss = TarDisClusterLoss(args, epoch, ca_t, target_target, tar_index, tar_cs, p_label_src, p_label_tar, th, emb=False)
        loss += weight_tar_cls * tardis_loss

    if args.dis_feat_tar:
    # Update target domain
        prob_pred = (1 + (f_t.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2)).pow(- (1) / 2)
        prob_pred_2 = (1 + (f_t_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2)).pow(- (1) / 2)

        tar_cluster_loss1 = TarDisClusterLoss(args, epoch, prob_pred, target_target, tar_index, tar_cs, p_label_src, p_label_tar, th, emb=True)
        loss += weight_tar_cluster * tar_cluster_loss1

        tar_cluster_loss2 = TarDisClusterLoss(args, epoch, prob_pred_2, target_target, tar_index, tar_cs, p_label_src, p_label_tar, th, emb=True)
        loss += weight_tar_cluster * tar_cluster_loss2


    # model forward on source
    f_s, f_s_2, ca_s = model(input_source_var)
    if args.domain_adv:
        d_s_loss = CondDiscriminatorLoss(epoch, ca_s, index, src_cs, src=True, dis_cls=False)
        loss += weight_dis * d_s_loss

    if args.dis_src:
        src_dis_loss = SrcClassifyLoss(args, epoch, ca_s, target_source, index, src_cs, p_label_src, p_label_tar, emb=False)
        loss += src_dis_loss

    if args.dis_feat_src:
        prob_pred = (1 + (f_s.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2)).pow(- (1) / 2)
        loss += weight_src_cluster * SrcClassifyLoss(args, epoch, prob_pred, target_source, index, src_cs, p_label_src, p_label_tar, emb=True)

        prob_pred_2 = (1 + (f_s_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2)).pow(- (1) / 2)
        loss += weight_src_cluster * SrcClassifyLoss(args, epoch, prob_pred_2, target_source, index, src_cs, p_label_src, p_label_tar, emb=True)

    losses.update(loss.data.item(), input_target.size(0))
    # loss backward and network update
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # Update join classifier & discriminator
    loss = 0

    f_t, f_t_2, ca_t = model(input_target_var)
    if args.domain_adv:
        d_t_loss = CondDiscriminatorLoss(epoch, ca_t, tar_index, tar_cs, src=False, dis_cls=True)
        loss += weight_dis * d_t_loss

    if args.dis_tar:
        tardis_loss = TarDisClusterLoss(args, epoch, ca_t, target_target, tar_index, tar_cs, p_label_src, p_label_tar, th, emb=False)
        loss += weight_tar_cls * tardis_loss


    if args.dis_feat_tar:
        prob_pred = (1 + (f_t.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2)).pow(- (1) / 2)
        prob_pred_2 = (1 + (f_t_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2)).pow(- (1) / 2)

        tar_cluster_loss1 = TarDisClusterLoss(args, epoch, prob_pred, target_target, tar_index, tar_cs, p_label_src, p_label_tar, th, emb=True)
        loss += weight_tar_cluster * tar_cluster_loss1

        tar_cluster_loss2 = TarDisClusterLoss(args, epoch, prob_pred_2, target_target, tar_index, tar_cs, p_label_src, p_label_tar, th, emb=True)
        loss += weight_tar_cluster * tar_cluster_loss2

    f_s, f_s_2, ca_s = model(input_source_var)
    # model forward on source
    if args.domain_adv:
        d_s_loss = CondDiscriminatorLoss(epoch, ca_s, index, src_cs, src=True, dis_cls=True)
        loss += weight_dis * d_s_loss

    if args.dis_src:
        src_dis_loss = SrcClassifyLoss(args, epoch, ca_s, target_source, index, src_cs, p_label_src, p_label_tar, emb=False)
        loss += src_dis_loss

    if args.dis_feat_src:
        prob_pred = (1 + (f_s.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2)).pow(- (1) / 2)
        loss += weight_src_cluster * SrcClassifyLoss(args, epoch, prob_pred, target_source, index, src_cs, p_label_src, p_label_tar, emb=True)

        prob_pred_2 = (1 + (f_s_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2)).pow(- (1) / 2)
        loss += weight_src_cluster * SrcClassifyLoss(args, epoch, prob_pred_2, target_source, index, src_cs, p_label_src, p_label_tar, emb=True)

    prec1_s = accuracy(ca_s.data, target_source, topk=(1,))[0]
    top1_source.update(prec1_s.item(), input_source.size(0))

    model.zero_grad()
    loss.backward()
    optimizer_cls.step()

    batch_time.update(time.time() - end)
    if itern % args.print_freq == 0:
        print('Train - epoch [{0}/{1}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'S@1 {s_top1.val:.3f} ({s_top1.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
               epoch, args.epochs, batch_time=batch_time,
               data_time=data_time, s_top1=top1_source, loss=losses))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write("\nTrain - epoch: %d, top1_s acc: %3f, loss: %4f" % (epoch, top1_source.avg, losses.avg))
        log.close()

    return train_loader_source_batch, train_loader_target_batch

def CondDiscriminatorLoss(epoch, output, index, cs, src=True, dis_cls=True):

    prob_p = F.softmax(output, dim=1)

    prob_p_dis = prob_p[:, -1].unsqueeze(1)
    prob_p_class = prob_p[:, :-1] / (1-prob_p_dis)

    weights = cs[index]

    if dis_cls:
        if src:
            loss_d = - ((weights) * ((1-prob_p_dis).log())).mean()
        else:
            if epoch <= 5:
                weights.fill_(1)
            loss_d = - ((weights) * (prob_p_dis.log())).mean()

    else:
        if src:
            loss_d = - ((weights) * (prob_p_dis.log())).mean()
        else:
            if epoch <= 5:
                weights.fill_(1)
            loss_d = - ((weights) * ((1-prob_p_dis).log())).mean()


    return loss_d


def TarDisClusterLoss(args, epoch, output, target, index, tar_cs, p_label_src, p_label_tar, th, emb):

    prob_p = F.softmax(output, dim=1)
    pos_loss = 0
    neg_loss = 0

    if not emb:
        prob_p_dis = prob_p[:, -1].unsqueeze(1)
        prob_p_class = prob_p[:, :-1]
        prob_p_class = prob_p_class / (1-prob_p_dis)

    else:
        prob_p_class = prob_p


    prob_q2 = prob_p_class / prob_p_class.sum(0, keepdim=True).pow(0.5)
    prob_q2 /= prob_q2.sum(1, keepdim=True)
    prob_q = prob_q2

    tar_weights = tar_cs[index.cuda()]
    pos_mask = torch.where(tar_weights >= th[target], 1, 0)

    class_weight = torch.exp(p_label_src) / torch.exp(p_label_tar)

    if epoch < 1:
        pos_loss = 0
        neg_loss = 0

    elif args.conf_pseudo_label:

        if epoch <= args.warmup:
            class_weight.fill_(1)
            tar_weights.fill_(1)

        elif len(torch.unique(pos_mask)) == 2:
            pos_loss = - (tar_weights[pos_mask==1] * (class_weight * prob_q[pos_mask==1] * prob_p_class[pos_mask==1].log()).sum(1)).mean()
            neg_loss = - ((1-tar_weights[pos_mask==0]) * (class_weight * prob_q[pos_mask==0] * (1-prob_p_class[pos_mask==0]).log()).sum(1)).mean()

        else:
            pos_loss = - (tar_weights[pos_mask==1] * (class_weight * prob_q[pos_mask==1] * prob_p_class[pos_mask==1].log()).sum(1)).mean()
            neg_loss = 0

    else:
        pos_loss = - (tar_weights * (class_weight * prob_q * prob_p_class.log()).sum(1)).mean()
        neg_loss = 0

    return pos_loss + neg_loss

def SrcClassifyLoss(args, epoch, output, target, index, src_cs, p_label_src, p_label_tar, emb):

    prob_p = F.softmax(output, dim=1)

    if not emb:
        prob_p_dis = prob_p[:, -1].unsqueeze(1)
        prob_p_class = prob_p[:, :-1]
        prob_p_class = prob_p_class / (1-prob_p_dis)
    else:
        prob_p_class = prob_p

    prob_q = Variable(torch.cuda.FloatTensor(prob_p_class.size()).fill_(0))
    prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p_class.size(0), 1).cuda())

    src_weights = src_cs[index].cuda()
    class_weight = torch.exp(p_label_tar) / torch.exp(p_label_src)

    if epoch < 1:
        class_weight.fill_(1)

    pos_loss = - (src_weights * (class_weight * prob_q * prob_p_class.log()).sum(1)).mean()

    return pos_loss


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    total_vector = torch.FloatTensor(args.num_classes).fill_(0)
    correct_vector = torch.FloatTensor(args.num_classes).fill_(0)
    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        target = target.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        # forward
        with torch.no_grad():
            _, _, output = model(input_var)
            output = output[:, :-1]
            loss = criterion(output, target_var)

        # compute and record loss and accuracy
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        total_vector, correct_vector = accuracy_for_each_class(output.data, target, total_vector, correct_vector) # compute class-wise accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
# measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test on T test set - [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))

    acc_for_each_class = 100.0 * correct_vector / total_vector
    print(' * Test on T test set - Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n             Test on T test set - epoch: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" % (epoch, losses.avg, top1.avg, top5.avg))
    log.write("\nAcc for each class: ")
    for i in range(args.num_classes):
        if i == 0:
            log.write("%dst: %3f" % (i+1, acc_for_each_class[i]))
        elif i == 1:
            log.write(",  %dnd: %3f" % (i+1, acc_for_each_class[i]))
        elif i == 2:
            log.write(", %drd: %3f" % (i+1, acc_for_each_class[i]))
        else:
            log.write(", %dth: %3f" % (i+1, acc_for_each_class[i]))
    log.write("\n                          Avg. over all classes: %3f" % acc_for_each_class.mean())
    log.close()
    return top1.avg, acc_for_each_class

def validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, compute_cen=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # compute source class centroids
    source_features = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), 2048).fill_(0)
    source_features_2 = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), 2048//4).fill_(0)
    source_targets = torch.cuda.LongTensor(len(val_loader_source.dataset.imgs)).fill_(0)
    labels_src = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), args.num_classes).fill_(0)
    c_src = torch.cuda.FloatTensor(args.num_classes, 2048).fill_(0)
    c_src_2 = torch.cuda.FloatTensor(args.num_classes, 2048//4).fill_(0)
    count_s = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    if compute_cen:
        for i, (input, target, index) in enumerate(val_loader_source): # the iterarion in the source dataset
            input_var = Variable(input)
            target = target.cuda()
            with torch.no_grad():
                feature, feature_2, output = model(input_var)
                output = output[:, :-1]
            source_features[index.cuda(), :] = feature.data.clone()
            source_features_2[index.cuda(), :] = feature_2.data.clone()
            source_targets[index.cuda()] = target.clone()
            target_ = torch.cuda.FloatTensor(output.size()).fill_(0)
            target_.scatter_(1, target.unsqueeze(1), torch.ones(output.size(0), 1).cuda())
            labels_src[index.cuda(), :] = target_.clone()

            c_src += (feature.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
            c_src_2 += (feature_2.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
            count_s += target_.sum(0).unsqueeze(1)

    target_features = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), 2048).fill_(0)
    target_features_2 = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), 2048//4).fill_(0)
    target_targets = torch.cuda.LongTensor(len(val_loader_target.dataset.imgs)).fill_(0)
    pseudo_labels = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), args.num_classes).fill_(0)
    c_tar = torch.cuda.FloatTensor(args.num_classes, 2048).fill_(0)
    c_tar_2 = torch.cuda.FloatTensor(args.num_classes, 2048//4).fill_(0)
    count_t = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)

    total_vector = torch.FloatTensor(args.num_classes).fill_(0)
    correct_vector = torch.FloatTensor(args.num_classes).fill_(0)

    end = time.time()
    for i, (input, target, index) in enumerate(val_loader_target): # the iterarion in the target dataset
        data_time.update(time.time() - end)
        input_var = Variable(input)
        target = target.cuda()
        target_var = Variable(target)

        with torch.no_grad():
            feature, feature_2, output = model(input_var)
            output = output[:, :-1]
        target_features[index.cuda(), :] = feature.data.clone() # index:a tensor
        target_features_2[index.cuda(), :] = feature_2.data.clone()
        target_targets[index.cuda()] = target.clone()
        pseudo_labels[index.cuda(), :] = output.data.clone()

        if compute_cen: # compute target class centroids
            pred = output.data.max(1)[1]
            pred_ = torch.cuda.FloatTensor(output.size()).fill_(0)
            pred_.scatter_(1, pred.unsqueeze(1), torch.ones(output.size(0), 1).cuda())
            c_tar += (feature.unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
            c_tar_2 += (feature_2.unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
            count_t += pred_.sum(0).unsqueeze(1)

        # compute and record loss and accuracy
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        total_vector, correct_vector = accuracy_for_each_class(output.data, target, total_vector, correct_vector) # compute class-wise accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test on T training set - [{0}][{1}/{2}]\t'
                  'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'D {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'T@1 {tc_top1.val:.3f} ({tc_top1.avg:.3f})\t'
                  'T@5 {tc_top5.val:.3f} ({tc_top5.avg:.3f})\t'
                  'L {tc_loss.val:.4f} ({tc_loss.avg:.4f})'.format(
                   epoch, i, len(val_loader_target), batch_time=batch_time,
                   data_time=data_time, tc_top1=top1, tc_top5=top5, tc_loss=losses))

    # compute global class centroids
    c_srctar = torch.cuda.FloatTensor(args.num_classes, 2048).fill_(0)
    c_srctar_2 = torch.cuda.FloatTensor(args.num_classes, 2048//4).fill_(0)
    c_srctar = (c_src + c_tar) / (count_s + count_t)
    c_srctar_2 = (c_src_2 + c_tar_2) / (count_s + count_t)
    c_src /= count_s
    c_src_2 /= count_s
    c_tar /= (count_t + 1e-6)
    c_tar_2 /= (count_t + 1e-6)

    acc_for_each_class = 100.0 * correct_vector / total_vector

    print(' * Test on T training set - Prec@1 {tc_top1.avg:.3f}, Prec@5 {tc_top5.avg:.3f}'.format(tc_top1=top1, tc_top5=top5))

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\nTest on T training set - epoch: %d, tc_loss: %4f, tc_Top1 acc: %3f, tc_Top5 acc: %3f" % (epoch, losses.avg, top1.avg, top5.avg))

    target_preds = pseudo_labels.argmax(dim=1)

    log.write("\nAcc for each class: ")
    for i in range(args.num_classes):
        if i == 0:
            log.write("%dst: %3f" % (i+1, acc_for_each_class[i]))
        elif i == 1:
            log.write(",  %dnd: %3f" % (i+1, acc_for_each_class[i]))
        elif i == 2:
            log.write(", %drd: %3f" % (i+1, acc_for_each_class[i]))
        else:
            log.write(", %dth: %3f" % (i+1, acc_for_each_class[i]))
    log.write("\n                          Avg. over all classes: %3f" % acc_for_each_class.mean())
    log.close()

    return acc_for_each_class.mean(), c_src, c_src_2, c_tar, c_tar_2, c_srctar, c_srctar_2, source_features, source_features_2, source_targets, target_features, target_features_2, target_targets, pseudo_labels, labels_src, target_preds

def source_select(source_features, source_targets, target_features, pseudo_labels, train_loader_source, epoch, cen, args):
    # compute source weights
    source_cos_sim_temp = source_features.unsqueeze(1) * cen.unsqueeze(0)
    source_cos_sim = 0.5 * (1 + source_cos_sim_temp.sum(2) / (source_features.norm(2, dim=1, keepdim=True) * cen.norm(2, dim=1, keepdim=True).t() + 1e-6))
    src_cs = torch.gather(source_cos_sim, 1, source_targets.unsqueeze(1)).squeeze(1)

    del source_cos_sim_temp
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return src_cs

def kernel_k_means(target_features, target_targets, pseudo_labels, train_loader_target, epoch, model, args, best_prec, change_target=True):
    # define kernel k-means clustering
    kkm = KernelKMeans(n_clusters=args.num_classes, max_iter=args.cluster_iter, random_state=0, kernel=args.cluster_kernel, gamma=args.gamma, verbose=1)
    kkm.fit(np.array(target_features.cpu()), initial_label=np.array(pseudo_labels.max(1)[1].long().cpu()), true_label=np.array(target_targets.cpu()), args=args, epoch=epoch)

    idx_sim = torch.from_numpy(kkm.labels_)
    c_tar = torch.cuda.FloatTensor(args.num_classes, target_features.size(1)).fill_(0)
    count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    for i in range(target_targets.size(0)):
        c_tar[idx_sim[i]] += target_features[i]
        count[idx_sim[i]] += 1
        if change_target:
            train_loader_target.dataset.tgts[i] = idx_sim[i].item()
    c_tar /= (count + 1e-6)

    prec1 = kkm.prec1_
    is_best = prec1 > best_prec
    if is_best:
        best_prec = prec1

    del target_features
    del target_targets
    del pseudo_labels
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return best_prec, c_tar

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    lr = args.lr / math.pow((1 + 10 * epoch / args.epochs), 0.75)
    for param_group in optimizer.param_groups:
       if param_group['name'] == 'conv':
           param_group['lr'] = lr
       elif param_group['name'] == 'ca_cl':
           param_group['lr'] = lr * 10
       elif param_group['name'] == 'cen':
           param_group['lr'] = lr * 10

       else:
           raise ValueError('The required parameter group does not exist.')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.view(-1).float().cpu() == target.float().cpu()
    for i in range(batch_size):
        total_vector[target[i]] += 1
        correct_vector[torch.LongTensor([target[i]])] += correct[i]

    return total_vector, correct_vector

