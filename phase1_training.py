import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import dataset
from datetime import datetime
import random
import numpy as np
import copy
import torchvision.models as models


import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='dataset for training')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=257, help='number of epochs to train (default: 10)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--la', type=float, default = 0.6)
parser.add_argument('--lc', type=float, default = 0.1)
parser.add_argument('--lr', type=float, default = 0.01)
parser.add_argument('--pgd_params', default = '8_4_10')
parser.add_argument('--baseline_model_pth', default = '.')
parser.add_argument('--train_pth', default = '.')
parser.add_argument('--test_pth', default = '.')
parser.add_argument('--save_pth', default = '.')
args = parser.parse_args()

print('==> Preparing data..')
if args.type == 'cifar10':
    trainloader, test_loader = dataset.get10(batch_size=args.batch_size)
    load_file = torch.load(args.baseline_model_pth, map_location='cpu')
    model = models.vgg11_bn()
    model.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=256, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=256, out_features=10, bias=True))

if args.type == 'cifar100':
    trainloader, test_loader = dataset.get100(batch_size=args.batch_size)
    load_file = torch.load(args.baseline_model_pth, map_location='cpu')
    model = models.vgg16_bn()
    model.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=256, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=256, out_features=100, bias=True))


if args.type == 'tinyimagenet':
    trainloader, testloader = dataset.tinyimagenet(batch_size=args.batch_size)

    net = models.vgg16_bn()
    print(net)
    net.classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=1024, out_features=512, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=200, bias=True))

model = torch.nn.DataParallel(model)
try:
    model.load_state_dict(load_file.state_dict())
except:
    model.load_state_dict(load_file)
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr)

best_acc, old_file = 0, None
best_loss = 1000
best_sep = 0
lambda_adv = torch.tensor([args.la]).cuda()
lambda_clean = torch.tensor([args.lc]).cuda()
t_begin = time.time()
mseloss = torch.nn.MSELoss()
celoss = torch.nn.CrossEntropyLoss()

train_loader = torch.load(args.train_pth)
test_loader_adv = torch.load(args.test_pth)

for epoch in range(args.epochs):
    print(f'epoch : {epoch}')
    model.train()
    random.shuffle(train_loader)
    print("training phase")
    for batch_idx, (data, target, y) in enumerate(train_loader):
        indx_target = target.clone()
        data, target, y = data.cuda(), target.cuda(), y.cuda()
        data, target, y  = Variable(data), Variable(target), Variable(y)
        optimizer.zero_grad()
        output = model(data)
        sum_of_I = model.module.features[0](data).abs().mean(dim=1).mean(dim=1).mean(dim=1)
        loss =  ((y)*mseloss(sum_of_I,lambda_adv))+((1-y)*mseloss(sum_of_I,lambda_clean))
        loss.sum().backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:

            print(f'adv_energy_loss : {(y*mseloss(sum_of_I,lambda_adv)).sum()} clean_energy_loss : {((1-y)*mseloss(sum_of_I,lambda_clean)).sum()}')

    elapse_time = time.time() - t_begin
    speed_epoch = elapse_time / (epoch + 1)
    speed_batch = speed_epoch / 50000
    eta = speed_epoch * args.epochs - elapse_time

    if epoch % args.test_interval == 0:
        model.eval()
        test_loss = 0
        correct = 0
        energy_a_loss = 0
        energy_avg = 0
        energy_mean_a = 0
        print("testing phase adv")
        for i, (data, target) in enumerate(test_loader_adv):
            indx_target = target.clone()
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                output = model(data)
                energy_a = model.module.features[0](data).abs().mean(dim=1).mean(dim=1).mean(dim=1)
                energy_mean_a += energy_a.mean()
                energy_a_loss += mseloss(energy_a,lambda_adv)

        energy_a_loss = energy_a_loss/len(test_loader_adv)
        mean_soi_a = energy_mean_a/len(test_loader_adv)

        print(f'energy loss a : {energy_a_loss} mean_soi_a {mean_soi_a}')

        energy_c = 0
        test_loss = 0
        correct = 0
        energy_c_loss = 0
        energy_mean_c = 0
        print("testing phase clean")
        for i, (data, target) in enumerate(test_loader):
            indx_target = target.clone()
            # if args.cuda:
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                output = model(data)
                energy_c = model.module.features[0](data).abs().mean(dim=1).mean(dim=1).mean(dim=1)
                energy_mean_c += energy_c.mean()

                energy_c_loss += mseloss(energy_c, lambda_clean)

        energy_c_loss = energy_c_loss/len(test_loader)
        mean_soi_c = energy_mean_c / len(test_loader)
        print(f'energy loss c : {energy_c_loss} mean_soi_c {mean_soi_c}')

        loss_energy = energy_c_loss + energy_a_loss
        print(f'total loss: {loss_energy}')
        print(f'sep: {mean_soi_a - mean_soi_c}')
        if (mean_soi_a - mean_soi_c) > best_sep:
            new_file = args.save_pth+'/'+args.type+'_'+args.pgd_params+'.pth'
            print(f'saving at {new_file}')
            torch.save(model, new_file)
            best_sep = mean_soi_a - mean_soi_c


