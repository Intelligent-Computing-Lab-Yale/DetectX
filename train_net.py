import argparse
import os
import time
# from utee import misc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from utee import make_path
import dataset
# from cifar import model
# from utee import wage_util
from datetime import datetime
# from utee import wage_quantizer
import random
import numpy as np
import copy
import torchvision.models as models


import torch.nn.functional as F
# import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='dataset for training')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=257, help='number of epochs to train (default: 10)')
# parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
# parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
# parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
# parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
# parser.add_argument('--wl_weight', type = int, default=2)
# parser.add_argument('--wl_grad', type = int, default=8)
# parser.add_argument('--wl_activate', type = int, default=8)
# parser.add_argument('--wl_error', type = int, default=8)
# parser.add_argument('--inference', default=0)
# parser.add_argument('--onoffratio', default=10)
# parser.add_argument('--cellBit', default=1)
# parser.add_argument('--subArray', default=128)
# parser.add_argument('--ADCprecision', default=5)
# parser.add_argument('--vari', default=0)
# parser.add_argument('--t', default=0)
# parser.add_argument('--v', default=0)
# parser.add_argument('--detect', default=0)
# parser.add_argument('--target', default=0)

parser.add_argument('--la', type=float, default = 0.6)
parser.add_argument('--lc', type=float, default = 0.1)
parser.add_argument('--lr', type=float, default = 0.01)
parser.add_argument('--pgd_params', default = '0.06_0.03_10')
parser.add_argument('--baseline_model_pth', default = '.')
parser.add_argument('--train_pth', default = '.')
parser.add_argument('--test_pth', default = '.')
parser.add_argument('--save_pth', default = '.')
args = parser.parse_args()

print('==> Preparing data..')
if args.type == 'cifar10':
    trainloader, test_loader = dataset.get10(batch_size=args.batch_size)
    # net = cifar100_net.Net()
    load_file = torch.load(args.baseline_model_pth, map_location='cpu')
    model = models.vgg11_bn()
    model.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=256, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=256, out_features=10, bias=True))

if args.type == 'cifar100':
    trainloader, test_loader = dataset.get100(batch_size=args.batch_size)
    # load_file = torch.load('cifar100_baseline_vgg16_bn.pth', map_location='cpu')
    load_file = torch.load(args.baseline_model_pth, map_location='cpu')

    ##################################    VGG16
    model = models.vgg16_bn()
    model.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=256, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=256, out_features=100, bias=True))

    ##################################    ResNet18
    # net = models.resnet18()
    # net.avgpool = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
    # net.fc = nn.Linear(in_features=512, out_features=1000, bias=True)
    # net.classifier = nn.Sequential(nn.Linear(in_features=1000, out_features=512, bias=True),
    #                                nn.ReLU(inplace=True), nn.Dropout(p=0.5),
    #                                nn.Linear(in_features=512, out_features=256, bias=True),
    #                                nn.ReLU(inplace=True), nn.Dropout(p=0.5),
    #                                nn.Linear(in_features=256, out_features=100, bias=True))

# if args.type == 'tinyimagenet':
#     trainloader, testloader = dataset.tinyimagenet(batch_size=args.batch_size)
#
#     ##################################    VGG16
#     net = models.vgg16_bn()
#     print(net)
#     net.classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
#     nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=1024, out_features=512, bias=True),
#     nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=200, bias=True))

model = torch.nn.DataParallel(model)
try:
    model.load_state_dict(load_file.state_dict())
except:
    model.load_state_dict(load_file)
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr)

# decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
# logger('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
best_loss = 1000
best_sep = 0
lambda_adv = torch.tensor([args.la]).cuda()
lambda_clean = torch.tensor([args.lc]).cuda()
t_begin = time.time()
# grad_scale = args.grad_scale
mseloss = torch.nn.MSELoss()
celoss = torch.nn.CrossEntropyLoss()

train_loader = torch.load(args.train_pth) #('./tinyimagenet/adv_train_set_tinyimagenet_pgd_e=0.125_ADC4_cellBit4_a=0.06_n=10', map_location = 'cpu')
test_loader_adv = torch.load(args.test_pth)

# train_loader = torch.load('./cifar100/adv_train_set_cifar100_pgd_e=0.25_ADC4_cellBit4_a=0.03_n=10') #('./tinyimagenet/adv_train_set_tinyimagenet_pgd_e=0.125_ADC4_cellBit4_a=0.06_n=10', map_location = 'cpu')
# test_loader_adv = torch.load('./cifar100/adv_test_set_cifar100_pgd_e=0.25_a=0.03_n=10')

# train_loader = torch.load('./cifar10_new/adv_train_set_cifar10_pgd_e=0.125_ADC4_cellBit4_a=0.06_n=10') #('./tinyimagenet/adv_train_set_tinyimagenet_pgd_e=0.125_ADC4_cellBit4_a=0.06_n=10', map_location = 'cpu')
# test_loader_adv = torch.load('./cifar10_new/adv_test_set_cifar10_pgd_e=0.125_a=0.06_n=10') #('./tinyimagenet/pgd_test_data/adv_test_set_tinyimagenet_pgd_e=0.125_a=0.06_n=10', map_location='cpu') #torch.load('./baseline/adv_test_set_cifar10_pgd_e=16_255_ADC4_cellBit4_a=8', map_location = 'cpu')
# train_loader = torch.load('./baseline/adv_train_set_cifar10_fgsm_e=16_255_ADC4_cellBit4')
# test_loader_adv = torch.load('./baseline/adv_test_set_cifar10_fgsm_e=16_255_ADC4_cellBit4')

for epoch in range(args.epochs):
    print(f'epoch : {epoch}')
    model.train()
    random.shuffle(train_loader)
    print("training phase")
    for batch_idx, (data, target, y) in enumerate(train_loader):
        # print(batch_idx)
        indx_target = target.clone()
        # if args.cuda:
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
    # print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
    #     elapse_time, speed_epoch, speed_batch, eta))

    # misc.model_save(model, os.path.join(args.logdir, 'latest.pth'))

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
            # if args.cuda:
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

            # new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
            new_file = args.save_pth+'/'+args.type+'_'+args.pgd_params+'.pth'
            print(f'saving at {new_file}')
            # misc.model_save(model, new_file, old_file=old_file, verbose=True)
            torch.save(model, new_file)
            # best_loss = loss_energy
            best_sep = mean_soi_a - mean_soi_c
            # old_file = new_file


