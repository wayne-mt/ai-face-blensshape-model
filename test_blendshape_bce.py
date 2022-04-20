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
import scipy
from torch.nn.modules.loss import  MultiLabelSoftMarginLoss as MultiLabelReg




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
    ###load the mode and head paramneters


    # model.load_state_dict(torch.load("checkpoints_blend/resnet18_190.pth"))
    # metric_fc.load_state_dict(torch.load("checkpoints_blend/resnet18-head_190.pth"))

    model_param = model.state_dict()
    head_param = metric_fc.state_dict()

    # print(model_param.keys())
    pretrained_model_dict = torch.load("checkpoints_blend/resnet18_190.pth")
    pretrained_head_dict = torch.load("checkpoints_blend/resnet18-head_190.pth")

    print(pretrained_head_dict.keys())
    for kk in model_param.keys():
        rep_kk = "module."+kk
        model_param[kk] = pretrained_model_dict[rep_kk]
    model.load_state_dict(model_param)

    print("model key {}".format(head_param.keys()))
    for kk in head_param.keys():
        rep_kk = "module."+kk
        head_param[kk] = pretrained_head_dict[rep_kk]
    metric_fc.load_state_dict(head_param)


    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)


    start = time.time()

    model.eval()
    accuarcy_bin=np.zeros((61,))
    tp=np.zeros((61,))
    fp=np.zeros((61,))
    fn=np.zeros((61,))
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

        output = output.data.cpu().numpy()
        label = label.data.cpu().numpy()
        print("predicted value v.s. gt values")
        ii = 0
        for a,b in zip(output[0,:], label[0,:]):
            print(" {}:{},{} ".format(ii, 1/(1 + np.exp(-a)), b))
            if b==1 and 1/(1 + np.exp(-a))>0.5:
                tp[ii]+=1
            elif b==0 and 1/(1 + np.exp(-a))>0.5:
                fp[ii]+=1
            elif b == 1 and 1 / (1 + np.exp(-a)) < 0.5:
                fn[ii] += 1
            ii+=1
        print("\n")
        # print(output)
        # print(label)
        # acc = np.mean((output == label).astype(int))
        speed = opt.print_freq / (time.time() - start)
        time_str = time.asctime(time.localtime(time.time()))
        # print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
        print('validation results: train epoch {} iter {} {} iters/s loss {}'.format(time_str, ii, speed, loss.item()))
        start = time.time()

    for i in range(61):
        print("metric detail TP {} FP {} FN {} Recall {} Precision {} at score==0.5 ".format(tp[i], fp[i], fn[i], float(tp[i])/(tp[i] + fn[i]), float(tp[i])/(tp[i] + fp[i] )))