'''Initialize the network architecture'''

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import adv_attacks

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import dataset
import torchvision.models as models



parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='dataset for training')
parser.add_argument('--batch_size', type= int, default=200, help='batch size for training')
parser.add_argument('--path', default='.', help='path where phase2 model is saved')
parser.add_argument('--pgd_params', default='0.125,0.007,7', help='PGD attack params')
parser.add_argument('--adv_train', type= int, default=0, help='if 1, adversarial training else training on clean data only')
parser.add_argument('--net', default='vgg11bn', help='type of net used')
parser.add_argument('--model_adv_gen', default='.', help='model for adversarial data gen')
parser.add_argument('--phase1_model', default='.', help='phase1 trained model')


args = parser.parse_args()

eps,alpha,steps = map(float, args.pgd_params.split(','))
print(eps,alpha,steps)
steps = int(steps)

print('==> Preparing data..')
if args.type == 'cifar10':
    trainloader, testloader = dataset.get10(batch_size=args.batch_size)
    net = models.vgg16_bn()
    net.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=256, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=256, out_features=10, bias=True))

    if args.adv_train == 1:
        model_adv_gen = models.vgg16_bn()
        model_adv_gen.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                         nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                         nn.Linear(in_features=512, out_features=256, bias=True),
                                         nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                         nn.Linear(in_features=256, out_features=10, bias=True))
        model_adv_gen = model_adv_gen.cuda()
        model_adv_gen = torch.nn.DataParallel(model_adv_gen)
        try:
            model_adv_gen.load_state_dict(torch.load(args.model_adv_gen).state_dict())
        except:
            model_adv_gen.load_state_dict(torch.load(args.model_adv_gen))

    net = net.cuda()
    net = torch.nn.DataParallel(net)
    try:
        net.load_state_dict(torch.load(args.phase1_model).state_dict())
    except:
        net.load_state_dict(torch.load(args.phase1_model))

    print(net)

if args.type == 'cifar100':
    trainloader, testloader = dataset.get100(batch_size=args.batch_size)
    net = models.vgg16_bn()
    net.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                     nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                     nn.Linear(in_features=512, out_features=256, bias=True),
                                     nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                     nn.Linear(in_features=256, out_features=100, bias=True))
    if args.adv_train == 1:
        model_adv_gen = models.vgg16_bn()
        model_adv_gen.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                         nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                         nn.Linear(in_features=512, out_features=256, bias=True),
                                         nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                         nn.Linear(in_features=256, out_features=100, bias=True))
        model_adv_gen = model_adv_gen.cuda()
        model_adv_gen = torch.nn.DataParallel(model_adv_gen)
        try:
            model_adv_gen.load_state_dict(torch.load(args.model_adv_gen).state_dict())
        except:
            model_adv_gen.load_state_dict(torch.load(args.model_adv_gen))

    net = net.cuda()
    net = torch.nn.DataParallel(net)
    try:
        net.load_state_dict(torch.load(args.phase1_model).state_dict())
    except:
        net.load_state_dict(torch.load(args.phase1_model))

if args.type == 'tinyimagenet':
    trainloader, testloader = dataset.tinyimagenet(batch_size=args.batch_size)

    net = models.vgg16_bn()
    print(net)
    net.classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=1024, out_features=512, bias=True),
    nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=200, bias=True))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    try:
        net.load_state_dict(torch.load(args.phase1_model).state_dict())
    except:
        net.load_state_dict(torch.load(args.phase1_model))

    if args.adv_train == 1:
        model_adv_gen = models.vgg16_bn()
        model_adv_gen.classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
                                       nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                       nn.Linear(in_features=1024, out_features=512, bias=True),
                                       nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                       nn.Linear(in_features=512, out_features=200, bias=True))
        model_adv_gen = model_adv_gen.cuda()
        model_adv_gen = torch.nn.DataParallel(model_adv_gen)
        try:
            model_adv_gen.load_state_dict(torch.load(args.model_adv_gen).state_dict())
        except:
            model_adv_gen.load_state_dict(torch.load(args.model_adv_gen))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

for param in net.parameters():
    param.requires_grad = True

net.module.features[0].weight.requires_grad = False
net.module.features[0].bias.requires_grad = False

'''train network'''

device = 'cuda'
best_acc = 0  # best test accuracy
num_epochs = 210

test_acc = []
for epoch in range(num_epochs):
    net.train()


    for batch_idx, (data, target) in enumerate(trainloader):
        indx_target = target.clone()
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # with torch.no_grad():
        if args.adv_train == 1:
            data_adv = adv_attacks.pgd_attack(model_adv_gen, data, target, eps, alpha, steps)
            data_aug = torch.cat((data, data_adv))
            target_aug = torch.cat((target, target))
        else:
            data_aug = data
            target_aug = target

        optimizer.zero_grad()
        output = net(data_aug)
        loss = criterion(output,target_aug)
        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0 and batch_idx > 0:
            pred = output.data.max(1)[1]
            if args.adv_train == 1:
                correct = pred.cpu().eq(torch.cat((indx_target,indx_target))).sum()
            else:
                correct = pred.cpu().eq(indx_target).sum()
            acc = float(correct) * 1.0 / len(data_aug)
            print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                loss.data, acc))

    if epoch % 1 == 0:
        print('testing phase')
        net.eval()
        test_loss = 0
        correct = 0
        correct_adv = 0
        for i, (data, target) in enumerate(testloader):
            indx_target = target.clone()
            clean_data = copy.deepcopy(data)
            data, target = data.cuda(), target.cuda()
            if args.adv_train == 1:
                data_adv = adv_attacks.pgd_attack(model_adv_gen, data, target, eps, alpha, steps)
                output = net(data_adv)
                pred0 = output.data.max(1)[1]  # get the index of the max log-probability
                correct_adv += pred0.cpu().eq(indx_target).sum()

            with torch.no_grad():
                c_data, target = Variable(clean_data), Variable(target)
                output = net(c_data)
                test_loss_i = criterion(output, target)
                test_loss += test_loss_i.data
                pred1 = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred1.cpu().eq(indx_target).sum()

        acc = 100. * correct / len(testloader.dataset)
        acc_adv = 100. * correct_adv / len(testloader.dataset)
        print(acc)
        print(acc_adv)

        if acc+acc_adv > best_acc:
            print('saving')
            if args.adv_train == 1:
                new_file = args.path+'/'+args.type+'_'+args.net+'_adv_train_phase2.pth'
            else:
                new_file = args.path + '/' + args.type + '_' + args.net + '_clean_phase2.pth'
            torch.save(net, new_file)
            best_acc = acc+acc_adv
