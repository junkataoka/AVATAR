import time
import torch
import os
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.kernel_kmeans import KernelKMeans
import gc
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchmetrics
import pandas as pd
from sklearn.manifold import TSNE

def train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, learn_cen, learn_cen_2, criterion_cons, optimizer, optimizer_cls, optimizer_cluster, itern, epoch, new_epoch_flag, src_cs, tar_cs, args, run, p_label_src, p_label_tar, th):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1_source = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    lam3 = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1 # penalty parameter
    lam2 = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1 # penalty parameter
    lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1 # penalty parameter

    weight_tar_cluster = lam
    weight_src_cluster = lam
    weight_tar_cls = lam
    weight_dis = lam

    adjust_learning_rate(optimizer, epoch, args) # adjust learning rate
    adjust_learning_rate(optimizer_cls, epoch, args) # adjust learning rate
    adjust_learning_rate(optimizer_cluster, epoch, args) # adjust learning rate

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
    target_source_var = Variable(target_source)
    target_target = target_target.cuda()
    input_target_var = Variable(input_target)
    target_target_var = Variable(target_target)


    loss = 0
    f_t, f_t_2, ca_t = model(input_target_var, 1)

    # Update target domain
    prob_pred = (1 + (f_t.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
    prob_pred_2 = (1 + (f_t_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)

    src_cs_const = Variable(torch.cuda.FloatTensor(src_cs.size()).fill_(src_cs[index].mean()))
    tar_cs_const = Variable(torch.cuda.FloatTensor(tar_cs.size()).fill_(src_cs[index].mean()))
    run["metrics/src_cs_mean"].log(src_cs.mean())
    run["metrics/src_cs_std"].log(src_cs.std())
    run["metrics/tar_cs_mean"].log(tar_cs.mean())
    run["metrics/tar_cs_std"].log(tar_cs.std())
    run["metrics/src_cs_min"].log(src_cs.min())
    run["metrics/src_cs_max"].log(src_cs.max())
    run["metrics/tar_cs_min"].log(tar_cs.min())
    run["metrics/tar_cs_max"].log(tar_cs.max())
    # tar_cs_const = Variable(torch.cuda.FloatTensor(tar_cs.size()).fill_(1))

    tar_cluster_loss1 = TarDisClusterLoss(args, epoch, prob_pred, target_target, tar_index, tar_cs, lam, p_label_src, p_label_tar, th, softmax=True, emb=True, em=False)
    run["metrics/tar_cluster_loss1"].log(tar_cluster_loss1)
    loss += weight_tar_cluster * tar_cluster_loss1

    tar_cluster_loss2 = TarDisClusterLoss(args, epoch, prob_pred_2, target_target, tar_index, tar_cs, lam, p_label_src, p_label_tar, th, softmax=True, emb=True, em=False)
    run["metrics/tar_cluster_loss2"].log(tar_cluster_loss2)
    loss += weight_tar_cluster * tar_cluster_loss2

    tardis_loss = TarDisClusterLoss(args, epoch, ca_t, target_target, tar_index, tar_cs, lam, p_label_src, p_label_tar, th, softmax=True, emb=False, em=False)
    loss += weight_tar_cls * tardis_loss

    d_t_loss = CondDiscriminatorLoss(args, epoch, ca_t, target_target, tar_index, tar_cs, lam, run, p_label_src, p_label_tar, fit=args.src_fit, src=False, dis_cls=False)
    loss += weight_dis * d_t_loss

    # model forward on source
    f_s, f_s_2, ca_s = model(input_source_var, 1)

    src_dis_loss = SrcClassifyLoss(args, epoch, ca_s, target_source, index, src_cs, lam, p_label_src, p_label_tar, softmax=True, emb=False)
    loss += src_dis_loss

    prob_pred = (1 + (f_s.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
    loss += weight_src_cluster * SrcClassifyLoss(args, epoch, prob_pred, target_source, index, src_cs, lam, p_label_src, p_label_tar, softmax=True,  emb=True)

    prob_pred_2 = (1 + (f_s_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
    loss += weight_src_cluster * SrcClassifyLoss(args, epoch, prob_pred_2, target_source, index, src_cs, lam, p_label_src, p_label_tar, softmax=True, emb=True)

    d_s_loss = CondDiscriminatorLoss(args, epoch, ca_s, target_source, index, src_cs, lam, run, p_label_src, p_label_tar, fit=args.src_fit, src=True, dis_cls=False)
    loss += weight_dis * d_s_loss

    class_weight = (p_label_src+0.5) / (p_label_tar+0.5)
    loss -= 1.0 * class_weight.mean()

    losses.update(loss.data.item(), input_target.size(0))
    # loss backward and network update
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # Update join classifier & discriminator
    loss = 0

    f_t, f_t_2, ca_t = model(input_target_var, 1)

    tardis_loss = TarDisClusterLoss(args, epoch, ca_t, target_target, tar_index, tar_cs, lam, p_label_src, p_label_tar, th, emb=False, em=False)
    loss += weight_tar_cls * tardis_loss
    run["metrics/tardis_loss"].log(tardis_loss)

    d_t_loss = CondDiscriminatorLoss(args, epoch, ca_t, target_target, tar_index, tar_cs, lam, run, p_label_src, p_label_tar, fit=args.src_fit, src=False, dis_cls=True)
    loss += weight_dis * d_t_loss
    run["metrics/d_t_loss"].log(d_t_loss)

    prob_pred = (1 + (f_t.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
    prob_pred_2 = (1 + (f_t_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)

    tar_cluster_loss1 = TarDisClusterLoss(args, epoch, prob_pred, target_target, tar_index, tar_cs, lam, p_label_src, p_label_tar, th, softmax=True, emb=True, em=False)
    run["metrics/tar_cluster_loss1"].log(tar_cluster_loss1)
    loss += weight_tar_cluster * tar_cluster_loss1

    tar_cluster_loss2 = TarDisClusterLoss(args, epoch, prob_pred_2, target_target, tar_index, tar_cs, lam, p_label_src, p_label_tar, th, softmax=True, emb=True, em=False)
    run["metrics/tar_cluster_loss2"].log(tar_cluster_loss2)
    loss += weight_tar_cluster * tar_cluster_loss2

    # model forward on source
    f_s, f_s_2, ca_s = model(input_source_var, 1)

    src_dis_loss = SrcClassifyLoss(args, epoch, ca_s, target_source, index, src_cs, lam, p_label_src, p_label_tar, fit=args.src_fit, emb=False)
    loss += src_dis_loss
    run["metrics/src_dis_loss"].log(src_dis_loss)

    prob_pred = (1 + (f_s.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
    loss += weight_src_cluster * SrcClassifyLoss(args, epoch, prob_pred, target_source, index, src_cs, lam, p_label_src, p_label_tar, softmax=True,  emb=True)

    prob_pred_2 = (1 + (f_s_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
    loss += weight_src_cluster * SrcClassifyLoss(args, epoch, prob_pred_2, target_source, index, src_cs, lam, p_label_src, p_label_tar, softmax=True, emb=True)

    d_s_loss = CondDiscriminatorLoss(args, epoch, ca_s, target_source, index, src_cs, lam, run, p_label_src, p_label_tar, fit=args.src_fit, src=True, dis_cls=True)
    run["metrics/d_s_loss"].log(d_s_loss)
    loss += weight_dis * d_s_loss

    class_weight = (p_label_src+0.5) / (p_label_tar+0.5)
    loss -= 1.0 * class_weight.mean()

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

def max_entropy_loss(args, epoch, s_output, t_output):

    temp = torch.cat([s_output, t_output], dim=1)
    prob = F.softmax(temp, dim=1)
    loss = -(prob * prob.log()).sum(1).mean()

    return loss


def CondDiscriminatorLoss(args, epoch, output, target, index, cs, lam, run, p_label_src, p_label_tar, softmax=True, fit=False, src=True, dis_cls=True):

    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(dim=1, keepdim=True)

    prob_p_dis = prob_p[:, -1].unsqueeze(1)
    prob_p_class = prob_p[:, :-1] / (1-prob_p_dis)


    weights = cs[index]

    entropy = -(prob_p_class * prob_p_class.log()).sum(1)
    entropy_weight = 1 + torch.exp(-entropy)
    entropy_weight.fill_(1)

    if dis_cls:
        if src:
            class_weight = (p_label_tar+0.5) / (p_label_src+0.5)
            loss_d = - ((weights * entropy_weight) * (class_weight[target]*(1-prob_p_dis).log())).mean()
            run["metrics/cls_src_prob"].log(prob_p_dis.mean())
        else:
            class_weight = (p_label_src+0.5) / (p_label_tar+0.5)
            loss_d = - ((weights * entropy_weight) * (class_weight[target]*prob_p_dis.log())).mean()
            run["metrics/cls_tar_prob"].log(prob_p_dis.mean())

    else:
        if src:
            class_weight = (p_label_tar+0.5) / (p_label_src+0.5)
            loss_d = - ((weights * entropy_weight) * (class_weight[target]*prob_p_dis.log())).mean()
            run["metrics/feat_src_prob"].log(prob_p_dis.mean())
        else:
            class_weight = (p_label_src+0.5) / (p_label_tar+0.5)
            loss_d = - ((weights * entropy_weight) * (class_weight[target]*(1-prob_p_dis).log())).mean()
            run["metrics/feat_tar_prob"].log(prob_p_dis.mean())
        # loss_d = - (src_weights * (prob_p_dis.log()).sum(1)).mean()
        # loss_d = - (weights * ((1-prob_p_dis).log()).sum(1)).mean()
        # prob_q2 = prob_p_class / prob_p_class.sum(0, keepdim=True).pow(0.5)
        # prob_q2 /= prob_q2.sum(1, keepdim=True)
        # prob_q = (1 - args.beta) * prob_q_class + args.beta * prob_q2
        # loss = - (src_weights * (prob_q* prob_p_class.log()).sum(1)).mean()


    return loss_d


def TarDisClusterLoss(args, epoch, output, target, index, tar_cs, lam, p_label_src, p_label_tar, th, softmax=True, em=False, emb=True):

    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)

    if not emb:
        prob_p_dis = prob_p[:, -1].unsqueeze(1)
        prob_p_class = prob_p[:, :-1]
        prob_p_class = prob_p_class / (1-prob_p_dis)

    else:
        prob_p_class = prob_p

    if em:
        prob_q = prob_p_class
    else:
        prob_q1 = Variable(torch.cuda.FloatTensor(prob_p_class.size()).fill_(0))
        prob_q1.scatter_(1, target.unsqueeze(1), torch.ones(prob_p_class.size(0), 1).cuda()) # assigned pseudo labels

        # if (epoch == 0) or args.ao:
        #     prob_q = prob_q1
        # else:
        #     prob_q2 = prob_p_class / prob_p_class.sum(0, keepdim=True).pow(0.5)
        #     prob_q2 /= prob_q2.sum(1, keepdim=True)
        #     prob_q = (1 - args.beta) * prob_q1 + args.beta * prob_q2
        prob_q2 = prob_p_class / prob_p_class.sum(0, keepdim=True).pow(0.5)
        prob_q2 /= prob_q2.sum(1, keepdim=True)
        prob_q = (1 - args.beta) * prob_q1 + args.beta * prob_q2


    tar_weights = tar_cs[index.cuda()]
    pos_mask = torch.where(tar_weights >= th[target], 1, 0)

    # class_weight = (p_label_src / p_label_tar) / (p_label_src / p_label_tar).sum()
    class_weight = (p_label_src+0.5) / (p_label_tar+0.5)

    if epoch < 1:
        class_weight.fill_(1)
        pos_loss = 0
        neg_loss = 0

    if len(torch.unique(pos_mask)) == 2:
        pos_loss = - (tar_weights[pos_mask==1] * (class_weight * prob_q[pos_mask==1] * prob_p_class[pos_mask==1].log()).sum(1)).mean()
        neg_loss = - ((1-tar_weights[pos_mask==0]) * (class_weight * prob_q[pos_mask==0] * (1-prob_p_class[pos_mask==0]).log()).sum(1)).mean()

    else:
        pos_loss = - (tar_weights[pos_mask==1] * (class_weight * prob_q[pos_mask==1] * prob_p_class[pos_mask==1].log()).sum(1)).mean()
        neg_loss = 0


    return pos_loss + neg_loss

def SrcClassifyLoss(args, epoch, output, target, index, src_cs, lam, p_label_src, p_label_tar, softmax=True, fit=False, emb=False):

    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)

    if not emb:
        prob_p_dis = prob_p[:, -1].unsqueeze(1)
        # prob_p_dis = torch.clamp(prob_p_dis, min=1e-3, max=0.999)
        prob_p_class = prob_p[:, :-1]
        # prob_p_class = torch.clamp(prob_p_class, min=1e-3, max=0.999)
        prob_p_class = prob_p_class / (1-prob_p_dis)
    else:
        prob_p_class = prob_p

    prob_q = Variable(torch.cuda.FloatTensor(prob_p_class.size()).fill_(0))
    prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p_class.size(0), 1).cuda())

    if fit:
        prob_q = (1 - prob_p) * prob_q + prob_p * prob_p

    src_weights = src_cs[index].cuda()
    class_weight = (p_label_tar+0.5) / (p_label_src+0.5)
    # class_weight = (p_label_src+0.5) / (p_label_tar+0.5)

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
            _, _, output = model(input_var, 1)
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
    if args.src.find('visda') != -1:
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
        return acc_for_each_class.mean()
    else:
        log.close()
        return top1.avg


def validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, run, compute_cen=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # compute source class centroids
    source_features = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), 2048).fill_(0)
    source_features_2 = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), args.num_neurons*4).fill_(0)
    source_targets = torch.cuda.LongTensor(len(val_loader_source.dataset.imgs)).fill_(0)
    labels_src = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), args.num_classes).fill_(0)
    c_src = torch.cuda.FloatTensor(args.num_classes, 2048).fill_(0)
    c_src_2 = torch.cuda.FloatTensor(args.num_classes, args.num_neurons*4).fill_(0)
    count_s = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    if compute_cen:
        for i, (input, target, index) in enumerate(val_loader_source): # the iterarion in the source dataset
            input_var = Variable(input)
            target = target.cuda()
            with torch.no_grad():
                feature, feature_2, output = model(input_var, 1)
                output = output[:, :-1]
            source_features[index.cuda(), :] = feature.data.clone()
            source_features_2[index.cuda(), :] = feature_2.data.clone()
            source_targets[index.cuda()] = target.clone()
            target_ = torch.cuda.FloatTensor(output.size()).fill_(0)
            target_.scatter_(1, target.unsqueeze(1), torch.ones(output.size(0), 1).cuda())
            labels_src[index.cuda(), :] = target_.clone()

            if args.cluster_method == 'spherical_kmeans':
                c_src += ((feature / feature.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * target_.unsqueeze(2)).sum(0)
                c_src_2 += ((feature_2 / feature_2.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * target_.unsqueeze(2)).sum(0)
            else:
                c_src += (feature.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
                c_src_2 += (feature_2.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
                count_s += target_.sum(0).unsqueeze(1)

    target_features = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), 2048).fill_(0)
    target_features_2 = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), args.num_neurons*4).fill_(0)
    target_targets = torch.cuda.LongTensor(len(val_loader_target.dataset.imgs)).fill_(0)
    pseudo_labels = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), args.num_classes).fill_(0)
    c_tar = torch.cuda.FloatTensor(args.num_classes, 2048).fill_(0)
    c_tar_2 = torch.cuda.FloatTensor(args.num_classes, args.num_neurons*4).fill_(0)
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
            feature, feature_2, output = model(input_var, 1)
            output = output[:, :-1]
        target_features[index.cuda(), :] = feature.data.clone() # index:a tensor
        target_features_2[index.cuda(), :] = feature_2.data.clone()
        target_targets[index.cuda()] = target.clone()
        pseudo_labels[index.cuda(), :] = output.data.clone()

        if compute_cen: # compute target class centroids
            pred = output.data.max(1)[1]
            pred_ = torch.cuda.FloatTensor(output.size()).fill_(0)
            pred_.scatter_(1, pred.unsqueeze(1), torch.ones(output.size(0), 1).cuda())
            if args.cluster_method == 'spherical_kmeans':
                c_tar += ((feature / feature.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
                c_tar_2 += ((feature_2 / feature_2.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
            else:
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
    c_srctar_2 = torch.cuda.FloatTensor(args.num_classes, args.num_neurons*4).fill_(0)
    if (args.cluster_method == 'spherical_kmeans'):
        c_srctar = c_src + c_tar
        c_srctar_2 = c_src_2 + c_tar_2
    else:
        c_srctar = (c_src + c_tar) / (count_s + count_t)
        c_srctar_2 = (c_src_2 + c_tar_2) / (count_s + count_t)
        c_src /= count_s
        c_src_2 /= count_s
        c_tar /= (count_t + args.eps)
        c_tar_2 /= (count_t + args.eps)

    acc_for_each_class = 100.0 * correct_vector / total_vector

    print(' * Test on T training set - Prec@1 {tc_top1.avg:.3f}, Prec@5 {tc_top5.avg:.3f}'.format(tc_top1=top1, tc_top5=top5))

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\nTest on T training set - epoch: %d, tc_loss: %4f, tc_Top1 acc: %3f, tc_Top5 acc: %3f" % (epoch, losses.avg, top1.avg, top5.avg))

    target_preds = pseudo_labels.argmax(dim=1)
    log_confusion_matrix(target_preds, target_targets, 31, "Target True VS Target Pred", run)
    tsne = TSNE(2)
    tsne_2 = TSNE(2)
    tsne_in = torch.cat([target_features, c_tar], dim=0)
    tsne_in_2 = torch.cat([target_features_2, c_tar_2], dim=0)
    tsne_proj = tsne.fit_transform(tsne_in.cpu().data.numpy())
    tsne_proj_2 = tsne_2.fit_transform(tsne_in_2.cpu().data.numpy())

    tsne_proj_cen = tsne_proj[-31:]
    tsne_proj_cen_2 = tsne_proj_2[-31:]
    tsne_proj = tsne_proj[:-31]
    tsne_proj_2 = tsne_proj_2[:-31]

    fig1, ax1 = plt.subplots(figsize=(12,12))
    fig2, ax2 = plt.subplots(figsize=(12,12))


    for g in range(31):
        ind = np.where(target_targets.cpu().data.numpy() == g)

        ax1.scatter(tsne_proj[ind, 0], tsne_proj[ind, 1],
                label=g,
                alpha=0.2)

        ax2.scatter(tsne_proj_2[ind, 0], tsne_proj_2[ind, 1],
                label=g,
                alpha=0.2)

    ax1.scatter(tsne_proj_cen[:, 0], tsne_proj_cen[:, 1], c="black")
    ax2.scatter(tsne_proj_cen_2[:, 0], tsne_proj_cen_2[:, 1], c="black")

    ax1.legend()
    ax2.legend()
    run["fig/target_tsne"].log(fig1)
    run["fig/target_tsne2"].log(fig2)
    plt.close(fig1)
    plt.close(fig2)

    if args.src.find('visda') != -1:
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
    else:
        log.close()
        return top1.avg, c_src, c_src_2, c_tar, c_tar_2, c_srctar, c_srctar_2, source_features, source_features_2, source_targets, target_features, target_features_2, target_targets, pseudo_labels, labels_src, target_preds

def log_confusion_matrix(target, label, num_classes, title, run):
    confusion_matrix = torchmetrics.functional.confusion_matrix(target, label, num_classes=num_classes)
    df_cm = pd.DataFrame(confusion_matrix.cpu().data.numpy(), index = range(num_classes), columns=range(num_classes))
    plt.figure(figsize = (12,12))
    fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
    plt.title(title)
    run[f"fig/{title}"].log(fig_)
    plt.close(fig_)

def source_select(source_features, source_targets, target_features, pseudo_labels, train_loader_source, epoch, cen, args):
    # compute source weights
    source_cos_sim_temp = source_features.unsqueeze(1) * cen.unsqueeze(0)
    source_cos_sim = 0.5 * (1 + source_cos_sim_temp.sum(2) / (source_features.norm(2, dim=1, keepdim=True) * cen.norm(2, dim=1, keepdim=True).t() + args.eps))
    src_cs = torch.gather(source_cos_sim, 1, source_targets.unsqueeze(1)).squeeze(1)

    # or hard source sample selection
    if args.src_hard_select:
        num_select_src_each_class = torch.cuda.LongTensor(args.num_classes).fill_(0)
        tao = 1 / (1 + math.exp(- args.tao_param * (epoch + 1))) - 0.01
        delta = np.log(args.num_classes) / 10
        indexes = torch.arange(0, source_features.size(0))

        target_kernel_sim = (1 + (target_features.unsqueeze(1) - cen.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
        if args.embed_softmax:
            target_kernel_sim = F.softmax(target_kernel_sim, dim=1)
        else:
            target_kernel_sim /= target_kernel_sim.sum(1, keepdim=True)
        _, pseudo_cat_dist = target_kernel_sim.max(dim=1)
        pseudo_labels_softmax = F.softmax(pseudo_labels, dim=1)
        _, pseudo_cat_std = pseudo_labels_softmax.max(dim=1)

        selected_indexes = []
        for c in range(args.num_classes):
            _, idxes = src_cs[source_targets == c].sort(dim=0, descending=True)

            temp1 = target_kernel_sim[pseudo_cat_dist == c].mean(dim=0)
            temp2 = pseudo_labels_softmax[pseudo_cat_std == c].mean(dim=0)
            temp1 = - (temp1 * ((temp1 + args.eps).log())).sum(0) # entropy 1
            temp2 = - (temp2 * ((temp2 + args.eps).log())).sum(0) # entropy 2
            if (temp1 > delta) and (temp2 > delta):
                tao -= 0.1
            elif (temp1 <= delta) and (temp2 <= delta):
                pass
            else:
                tao -= 0.05
            while 1:
                num_select_src_each_class[c] = (src_cs[source_targets == c][idxes] >= tao).float().sum()
                if num_select_src_each_class[c] > 0: # at least 1
                    selected_indexes.extend(list(np.array(indexes[source_targets == c][idxes][src_cs[source_targets == c][idxes] >= tao])))
                    break
                else:
                    tao -= 0.05

        train_loader_source.dataset.samples = []
        train_loader_source.dataset.tgts = []
        for idx in selected_indexes:
            train_loader_source.dataset.samples.append(train_loader_source.dataset.imgs[idx])
            train_loader_source.dataset.tgts.append(train_loader_source.dataset.imgs[idx][1])
        print('%d source instances have been selected at %d epoch' % (len(selected_indexes), epoch))
        print('Number of selected source instances each class: ', num_select_src_each_class)
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n~~~%d source instances have been selected at %d epoch~~~' % (len(selected_indexes), epoch))
        log.close()

        src_cs = torch.cuda.FloatTensor(len(train_loader_source.dataset.tgts)).fill_(1)

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
    c_tar /= (count + args.eps)

    prec1 = kkm.prec1_
    is_best = prec1 > best_prec
    if is_best:
        best_prec = prec1
        #torch.save(c_tar, os.path.join(args.log, 'c_t_kernel_kmeans_cluster_best.pth.tar'))
        #torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_kernel_kmeans_cluster_best.pth.tar'))

    del target_features
    del target_targets
    del pseudo_labels
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return best_prec, c_tar


def k_means(target_features, target_targets, train_loader_target, epoch, model, c, args, best_prec, change_target=True):
    batch_time = AverageMeter()

    c_tar = c.data.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        torch.cuda.empty_cache()
        dist_xt_ct_temp = target_features.unsqueeze(1) - c_tar.unsqueeze(0)
        dist_xt_ct = dist_xt_ct_temp.pow(2).sum(2)
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)
        prec1 = accuracy(-1 * dist_xt_ct.data, target_targets, topk=(1,))[0].item()
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1
            #torch.save(c_tar, os.path.join(args.log, 'c_t_kmeans_cluster_best.pth.tar'))
            #torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_kmeans_cluster_best.pth.tar'))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch %d, K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\nEpoch %d, K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        if args.src.find('visda') != -1:
            total_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            correct_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            total_vector_dist, correct_vector_dist = accuracy_for_each_class(-1 * dist_xt_ct.data, target_targets, total_vector_dist, correct_vector_dist)
            acc_for_each_class_dist = 100.0 * correct_vector_dist / (total_vector_dist + args.eps)
            log.write("\nAcc_dist for each class: ")
            for i in range(args.num_classes):
                if i == 0:
                    log.write("%dst: %3f" % (i+1, acc_for_each_class_dist[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i+1, acc_for_each_class_dist[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i+1, acc_for_each_class_dist[i]))
                else:
                    log.write(", %dth: %3f" % (i+1, acc_for_each_class_dist[i]))
            log.write("\n                          Avg_dist. over all classes: %3f" % acc_for_each_class_dist.mean())
        log.close()

        c_tar_temp = torch.cuda.FloatTensor(args.num_classes, c_tar.size(1)).fill_(0)
        count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
        for k in range(args.num_classes):
            c_tar_temp[k] += target_features[idx_sim.squeeze(1) == k].sum(0)
            count[k] += (idx_sim.squeeze(1) == k).float().sum()
        c_tar_temp /= (count + args.eps)

        if (itr == (args.cluster_iter - 1)) and change_target:
            for i in range(target_targets.size(0)):
                train_loader_target.dataset.tgts[i] = int(idx_sim[i])

        c_tar = c_tar_temp.clone()

        del dist_xt_ct_temp
        gc.collect()
        torch.cuda.empty_cache()

    del target_features
    del target_targets
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return best_prec, c_tar


def spherical_k_means(target_features, target_targets, train_loader_target, epoch, model, c, args, best_prec, change_target=True):
    batch_time = AverageMeter()

    c_tar = c.data.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        torch.cuda.empty_cache()
        dist_xt_ct_temp = target_features.unsqueeze(1) * c_tar.unsqueeze(0)
        dist_xt_ct = 0.5 * (1 - dist_xt_ct_temp.sum(2) / (target_features.norm(2, dim=1, keepdim=True) * c_tar.norm(2, dim=1, keepdim=True).t() + args.eps))
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)
        prec1 = accuracy(-1 * dist_xt_ct.data, target_targets, topk=(1,))[0].item()
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1
            #torch.save(c_tar, os.path.join(args.log, 'c_t_spherical_kmeans_cluster_best.pth.tar'))
            #torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_spherical_kmeans_cluster_best.pth.tar'))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch %d, Spherical K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\nEpoch %d, Spherical K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        if args.src.find('visda') != -1:
            total_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            correct_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            total_vector_dist, correct_vector_dist = accuracy_for_each_class(-1 * dist_xt_ct.data, target_targets, total_vector_dist, correct_vector_dist)
            acc_for_each_class_dist = 100.0 * correct_vector_dist / (total_vector_dist + args.eps)
            log.write("\nAcc_dist for each class: ")
            for i in range(args.num_classes):
                if i == 0:
                    log.write("%dst: %3f" % (i+1, acc_for_each_class_dist[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i+1, acc_for_each_class_dist[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i+1, acc_for_each_class_dist[i]))
                else:
                    log.write(", %dth: %3f" % (i+1, acc_for_each_class_dist[i]))
            log.write("\n                          Avg_dist. over all classes: %3f" % acc_for_each_class_dist.mean())
        log.close()
        c_tar_temp = torch.cuda.FloatTensor(args.num_classes, c_tar.size(1)).fill_(0)
        for k in range(args.num_classes):
            c_tar_temp[k] += (target_features[idx_sim.squeeze(1) == k] / (target_features[idx_sim.squeeze(1) == k].norm(2, dim=1, keepdim=True) + args.eps)).sum(0)

        if (itr == (args.cluster_iter - 1)) and change_target:
            for i in range(target_targets.size(0)):
                train_loader_target.dataset.tgts[i] = int(idx_sim[i])

        c_tar = c_tar_temp.clone()

        del dist_xt_ct_temp
        gc.collect()
        torch.cuda.empty_cache()

    del target_features
    del target_targets
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
    if args.lr_plan == 'step':
        exp = epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
        lr = args.lr * (0.1 ** exp)
    elif args.lr_plan == 'dao':
        lr = args.lr / math.pow((1 + 10 * epoch / args.epochs), 0.75)
    for param_group in optimizer.param_groups:
       if param_group['name'] == 'conv':
           param_group['lr'] = lr
       elif param_group['name'] == 'ca_cl':
           param_group['lr'] = lr
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

