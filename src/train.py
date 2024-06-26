import numpy as np
from torch.nn.init import xavier_uniform_
from helper import adjust_learning_rate
from losses import srcClassifyLoss, tarClassifyLoss, adversarialLoss
import math
import gc
import torch


def to_percent(temp, position):
    return '%1.0f' % (temp) + '%'

# model initialization 
def weight_init(m):
    class_name = m.__class__.__name__ 
    if class_name.find('Conv') != -1:
        xavier_uniform_(m.weight.data)
    if class_name.find('Linear') != -1:
        xavier_uniform_(m.weight.data)

# batch norm initialization
def batch_norm_init(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.reset_running_stats()



def train_avatar_batch(model, src_train_batch, tar_train_batch, 
                        src_train_dataloader, tar_train_dataloader, 
                        optimizer_dict, cur_epoch, logger, val_dict, args):

    try:
        (src_idx, src_input, src_target) = src_train_batch.__next__()[1]
    except StopIteration:
        src_train_batch = enumerate(src_train_dataloader)
        (src_idx, src_input, src_target) = src_train_batch.__next__()[1]

    try:
        (tar_idx, tar_input, tar_target) = tar_train_batch.__next__()[1]
    except StopIteration:
        tar_train_batch = enumerate(tar_train_dataloader)
        (tar_idx, tar_input, tar_target) = tar_train_batch.__next__()[1]

    src_input = src_input.float().cuda()
    src_target = src_target.unsqueeze(-1).long().cuda()
    tar_input = tar_input.float().cuda()
    tar_target = tar_target.unsqueeze(-1).long().cuda()

    # penalty parameter
    #lam = 2 / (1 + math.exp(-1 * 10 * cur_epoch / epochs)) - 1 
    loss_dict = {}
    p = float(cur_epoch) / 20
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    torch.cuda.empty_cache()
    model.train()
    src_class_prob, src_domain_prob, _ = model(src_input)
    tar_class_prob, tar_domain_prob, _ = model(tar_input)
    optimizer_dict["classifier"].zero_grad()
    adjust_learning_rate(optimizer_dict["classifier"], args.lr*10, cur_epoch, args.epochs) # adjust learning rate
    #src_class_prob, src_domain_prob, _ = model(src_input)
    #tar_class_prob, tar_domain_prob, _ = model(tar_input)

    loss_dict["src_loss_domain"] = adversarialLoss(args=args, epoch=cur_epoch, prob_p_dis=src_domain_prob, 
                                                    index=src_idx, weights_ord=val_dict["src_weights"], 
                                                    src=True, is_encoder=False)

    #assert math.isnan(loss_dict["src_loss_domain"]) == False

    loss_dict["tar_loss_domain"] = adversarialLoss(args=args, epoch=cur_epoch, prob_p_dis=tar_domain_prob, 
                                                    index=tar_idx, weights_ord=val_dict["tar_weights"], 
                                                    src=False, is_encoder=True)

    #assert math.isnan(loss_dict["tar_loss_domain"]) == False

    loss_dict["src_loss_class"] = srcClassifyLoss(src_class_prob, src_target, 
                                                  index=src_idx, weights_ord=val_dict["src_weights"])

    #assert math.isnan(loss_dict["src_loss_class"]) == False

    loss_dict["tar_loss_class"] = tarClassifyLoss(args=args, epoch=cur_epoch, tar_cls_p=tar_class_prob, 
                                                  target_ps_ord=val_dict["tar_label_kmeans"], 
                                                  index=tar_idx, weights_ord=val_dict["tar_weights"],
                                                  th=val_dict["th"])

    #assert torch.isnan(loss_dict["tar_loss_class"]) == False

    loss_dict["classifier_loss"]= alpha * (0.5 * (loss_dict["src_loss_domain"] + loss_dict["tar_loss_domain"]) + \
                                        loss_dict["src_loss_class"] + loss_dict["tar_loss_class"])

    loss_dict["classifier_loss"].backward()
    optimizer_dict["classifier"].step()

    torch.cuda.empty_cache()
    src_class_prob, src_domain_prob, _ = model(src_input)
    tar_class_prob, tar_domain_prob, _ = model(tar_input)
    adjust_learning_rate(optimizer_dict["encoder"], args.lr, cur_epoch, args.epochs) # adjust learning rate
    optimizer_dict["encoder"].zero_grad()

    loss_dict["src_loss_domain"] = adversarialLoss(args=args, epoch=cur_epoch, prob_p_dis=src_domain_prob, 
                                                    index=src_idx, weights_ord=val_dict["src_weights"].cuda(), 
                                                    src=True, is_encoder=True)


    loss_dict["tar_loss_domain"] = adversarialLoss(args=args, epoch=cur_epoch, prob_p_dis=tar_domain_prob, 
                                                    index=tar_idx, weights_ord=val_dict["tar_weights"].cuda(), 
                                                    src=False, is_encoder=True)


    loss_dict["src_loss_class"] = srcClassifyLoss(src_class_prob, src_target, 
                                                  index=src_idx, weights_ord=val_dict["src_weights"].cuda())


    loss_dict["tar_loss_class"] = tarClassifyLoss(args=args, epoch=cur_epoch, tar_cls_p=tar_class_prob, 
                                                  target_ps_ord=val_dict["tar_label_kmeans"].cuda(), 
                                                  index=tar_idx, weights_ord=val_dict["tar_weights"].cuda(),
                                                  th=val_dict["th"].cuda())


    loss_dict["encoder_loss"]= alpha * (0.5 * (loss_dict["src_loss_domain"] + loss_dict["tar_loss_domain"]) + \
                                            loss_dict["src_loss_class"] + loss_dict["tar_loss_class"])
    
    loss_dict["encoder_loss"].backward()
    optimizer_dict["encoder"].step()


    loss_dict["epoch"] = cur_epoch
    logger.log(loss_dict)
