from __future__ import print_function
import os
from data import DatasetBlend
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *

from torch.nn.modules.loss import  MultiLabelSoftMarginLoss as MultiLabelReg

def save_model(model, metric_fc, save_path, name, iter_cnt):
    save_name_base = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    save_name_head = os.path.join(save_path, name + '-head_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name_base)
    torch.save(metric_fc.state_dict(), save_name_head)

    return save_name_base, save_name_head


if __name__ == '__main__':

    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    train_dataset = DatasetBlend(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    val_dataset = DatasetBlend(opt.train_root, opt.val_list, phase='train', input_shape=opt.input_shape)
    valloader = data.DataLoader(val_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)



    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        # criterion = MultiLabelReg()
        # criterion = nn.SmoothL1Loss(beta=0.45)
        criterion = torch.nn.BCEWithLogitsLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        ###in the case of multi label regression, we just use linear reg for the first step
        metric_fc = MLReg(512, opt.num_classes)



    # view_model(model, opt.input_shape)
    # print(model)

    # model_param = model.state_dict()
    #
    # pretrained_model_dict = torch.load("checkpoints/resnet18_40.pth")
    # # pretrained_head_dict = torch.load("checkpoints_blend/resnet18-head_190.pth")
    # #
    # print(pretrained_model_dict.keys())
    # for kk in model_param.keys():
    #     rep_kk = "module." + kk
    #     model_param[kk] = pretrained_model_dict[rep_kk]
    # model.load_state_dict(model_param)

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    for i in range(opt.max_epoch):
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device)
            feature = model(data_input)
            ##check loss array shape
            # print("feature shape {} label shape {}".format(feature.shape, label.shape))
            output = metric_fc(feature)
            ##check loss array shape
            # print("output shape {} label shape {}".format(output.shape, label.shape))
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                label = label.data.cpu().numpy()
                # print(output[0,:])
                # print(label[0,:])
                # acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                # print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                print('{} train epoch {} iter {} {} iters/s loss {}'.format(time_str, i, ii, speed, loss.item(),
                                                                                   ))

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, metric_fc, opt.checkpoints_path, opt.backbone, i)

            model.eval()
            for ii, data in enumerate(valloader):
                data_input, label = data
                data_input = data_input.to(device)
                label = label.to(device)
                feature = model(data_input)
                ##check loss array shape
                # print("feature shape {} label shape {}".format(feature.shape, label.shape))
                output = metric_fc(feature)
                ##check loss array shape
                # print("output shape {} label shape {}".format(output.shape, label.shape))
                loss = criterion(output, label)
                iters = i * len(valloader) + ii

                if iters % opt.print_freq == 0:
                    output = output.data.cpu().numpy()
                    label = label.data.cpu().numpy()
                    # print(output)
                    # print(label)
                    # acc = np.mean((output == label).astype(int))
                    speed = opt.print_freq / (time.time() - start)
                    time_str = time.asctime(time.localtime(time.time()))
                    # print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                    print('validation results: {} train epoch {} iter {} {} iters/s loss {}'.format(time_str, i, ii, speed, loss.item()))
                    start = time.time()