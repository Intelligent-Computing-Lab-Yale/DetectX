import argparse
import torch
import copy
import dataset
import numpy as np
import adv_attacks
import os
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='cifar10|cifar100|TinyImagenet')
parser.add_argument('--a_type', default='fgsm')
parser.add_argument('--batch_size', type=int, default='200')
parser.add_argument('--lut_path', default='.')
parser.add_argument('--model_adv_gen', default='.')
parser.add_argument('--model_dual_phase', default='.')
parser.add_argument('--pgd_param', default='0.125,0.007,7')


args = parser.parse_args()

if args.type == 'cifar10':
    print('loading cifar10 dataset')
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1, train= True)
    import cifar10_net
    import cifar10_net_infer
    model = models.vgg11_bn()
    model.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=512, out_features=256, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=256, out_features=10, bias=True))
    model = torch.nn.DataParallel(model)
    try:
        model.load_state_dict(torch.load(args.model_adv_gen).state_dict())
    except:
        model.load_state_dict(torch.load(args.model_adv_gen))
    model_infer = models.vgg11_bn() #model.cifar10(args=args, logger=logger)
    model_infer.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=512, out_features=256, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=256, out_features=10, bias=True))
    model_infer = torch.nn.DataParallel(model_infer)
    try:
        model_infer.load_state_dict(torch.load(args.model_dual_phase).state_dict())
    except:
        model_infer.load_state_dict(torch.load(args.model_dual_phase))

    model = model.cuda()
    model_infer = model_infer.cuda()

elif args.type == 'cifar100':
    print('loading cifar100 dataset')
    train_loader, test_loader = dataset.get100(batch_size=args.batch_size)

    model = models.vgg16_bn() #tinyimagenet_net.ResNet().cuda()
    model.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=512, out_features=256, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=256, out_features=100, bias=True))

    model = torch.nn.DataParallel(model)
    try:
        model.load_state_dict(torch.load(args.model_adv_gen).state_dict())
    except:
        model.load_state_dict(torch.load(args.model_adv_gen))

    model_infer = models.vgg16_bn()
    model_infer.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=512, out_features=256, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=256, out_features=100, bias=True)) #cifar100_net_infer.Net().cuda()
    model_infer = torch.nn.DataParallel(model_infer)
    try:
        model_infer.load_state_dict(torch.load(args.model_dual_phase).state_dict())
    except:
        model_infer.load_state_dict(torch.load(args.model_dual_phase))

    model = model.cuda()
    model_infer = model_infer.cuda()

soi_clean_list = []
soi_adv_list = []
eps, alpha, steps = map(float, args.pgd_param.split(','))
steps = int(steps)
if args.a_type == 'pgd':
    print(f'PGD with eps = {eps}, alpha = {alpha}, steps = {steps}')
elif args.a_type == 'fgsm':
    print(f'FGSM with eps = {eps}')

for i, (data, labels_tru) in enumerate(test_loader):

    inp = data.cuda()

    soi = model_infer.module.features[0](inp).abs().mean(dim=1).mean(dim=1).mean(dim=1).detach()
    tuple_da = (soi, labels_tru)
    soi_clean_list.append(tuple_da)

for i, (data, labels_tru) in enumerate(test_loader):

    if args.a_type == 'pgd':
        inp_adv = adv_attacks.pgd_attack(model, data.cuda(), labels_tru.cuda(), eps, alpha, steps )#attack(data, labels_tru)
    elif args.a_type == 'fgsm':
        steps = int(steps)
        inp_adv = adv_attacks.fgsm_attack(data, eps, data_grad)
    # else:
        # print('clean')
    # inp_adv = data.cuda()

    # output,soi = model_infer(inp_adv)

    # soi = model_infer(inp_adv)
    soi_adv = model_infer.module.features[0](inp_adv).abs().mean(dim=1).mean(dim=1).mean(dim=1).detach()
    tuple_da = (soi_adv, labels_tru)

    # sum_of_I_list.append(tuple_da)
    soi_adv_list.append(tuple_da)

list_clean = []
for i in soi_clean_list:
    tuple_d = i[0]
    for j in tuple_d:
        list_clean.append(j.item())

list_adv = []
for i in soi_adv_list:
    tuple_d = i[0]
    for j in tuple_d:
        list_adv.append(j.item())


arr_clean = np.array(list_clean)
arr_adv = np.array(list_adv)

p = plt.hist(arr_clean, bins=300)
p1 = plt.hist(arr_adv, bins=p[1])

t_n = p[0]+p1[0]
t_n[t_n==0] = 1
prob_clean = np.true_divide(p[0], t_n)
prob_adv = np.true_divide(p1[0], t_n)
lut = [p[1], prob_clean]

file = args.lut_path+'/'+args.type+'_'+args.a_type+'_LUT'
print(f'saving LUT at {file}')
torch.save(lut, file)