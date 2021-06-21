import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pgd

import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
import dataset
from datetime import datetime
import torchvision.models as models
import os
import time
import numpy as np
import copy

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='dataset for training')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 200)')
parser.add_argument('--atype', default='fgsm', help='fgsm, pgd')
parser.add_argument('--eps', type=float, default=0, help='epsilon value')
parser.add_argument('--alpha', type=float, default=0, help='alpha value')
parser.add_argument('--steps', type=int, default=0, help='steps value')
parser.add_argument('--path', default='.', help='path where the adversarial test set should be stored')
parser.add_argument('--model_pth', default='.', help='model checkpoint file to use for adversarial data generation')
args = parser.parse_args()

if args.type == 'cifar10':
    print('loading cifar10 dataset')
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1, train= True)
elif args.type == 'cifar100':
    print('loading cifar100 dataset')
    train_loader, test_loader = dataset.get100(batch_size=args.batch_size, num_workers=1, train=True)
elif args.type == 'tinyimagenet':
    print('loading tinyimagenet dataset')
    train_loader, test_loader = dataset.tinyimagenet(batch_size=args.batch_size)

model_pth =  args.model_pth
print(f'model path for adversarial dataset generation: {model_pth}')

if args.type == 'cifar10':
    net = models.vgg11_bn()
    net.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=512, out_features=256, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=256, out_features=10, bias=True))
if args.type == 'cifar100':
    net = models.vgg16_bn()
    net.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=512, out_features=256, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=256, out_features=100, bias=True))

net = torch.nn.DataParallel(net)
try:
    net.load_state_dict(torch.load(model_pth).state_dict())
except:
    net.load_state_dict(torch.load(model_pth))
net = net.cuda()
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    if epsilon!=0:
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

from torch.autograd import Variable
def pgd_attack(net, device, testloader ):
    train_dataset = []
    for batch_idx, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels_tru = data

        # wrap them in Variable
        inp_var, true_label = Variable(inputs.cuda(), requires_grad=True), Variable(labels_tru.cuda()
                                                                                    , requires_grad=False)
        inp_adv = pgd.pgd_attack(net, inp_var, true_label, args.eps, args.alpha, args.steps)
        ifadv = torch.ones(200)
        ifnotadv = torch.zeros(200)
        adv_da_tuple = (inp_adv, labels_tru, ifadv)
        clean_da_tuple = (inp_var, labels_tru, ifnotadv)

        train_dataset.append(adv_da_tuple)
        train_dataset.append(clean_da_tuple)

    return train_dataset



def test_attack( model, device, testloader ):
  model.eval()
  train_dataset = []

  # Loop over all examples in test set
  for i, (data, target) in enumerate(testloader):

      # Send the data and label to the device
      data, target = data.to(device), target.to(device)

      # Set requires_grad attribute of tensor. Important for Attack
      data.requires_grad = True

      # Forward pass the data through the model
      output = model(data)

      # Calculate the loss
      loss = F.cross_entropy(output, target)

      # Zero all existing gradients
      model.zero_grad()

      # Calculate gradients of model in backward pass
      loss.backward()

      # Collect datagrad
      data_grad = data.grad.data

      # Call FGSM Attack
      perturbed_data = fgsm_attack(data, args.eps, data_grad)
      # pert = (perturbed_data-data).abs().mean()
      # print(pert)

      ifadv = torch.ones(200).cpu()
      ifnotadv = torch.zeros(200).cpu()
      adv_da_tuple = (perturbed_data.cpu(), target.cpu(),ifadv)

      clean_da_tuple = (data.cpu(), target.cpu(), ifnotadv)
      # print('hello')
      train_dataset.append(adv_da_tuple)

      train_dataset.append(clean_da_tuple)

  return train_dataset

if args.atype == 'fgsm':
    print('doing FGSM')
    train_dataset = test_attack( net, 'cuda', train_loader)
    file = args.path+'/adv_train_set_'+args.type+'_fgsm_e=' + str(args.eps)
    print(f'file saved at {file}')
    torch.save(train_dataset, file)

elif args.atype == 'pgd':
    print('doing PGD')
    train_dataset = pgd_attack(net, 'cuda', train_loader)
    file = args.path+'/adv_train_set_'+args.type+'_pgd_e=' + str(args.eps) + '_a='+str(args.alpha)+'_n='+str(args.steps)
    print(f'file saved at {file}')
    torch.save(train_dataset, file)
