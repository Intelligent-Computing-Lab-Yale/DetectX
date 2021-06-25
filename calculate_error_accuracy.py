
# import torchattacks
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



parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='cifar10|cifar100|TinyImagenet')
parser.add_argument('--a_type', default='fgsm')
parser.add_argument('--batch_size', type=int, default='200')
parser.add_argument('--soi_path', default='.')
parser.add_argument('--baseline_model', default='.')
parser.add_argument('--model_adv_gen', default='.')
parser.add_argument('--model_inference', default='.')
parser.add_argument('--lut_path', default='.')
parser.add_argument('--clean', type=int, default=1)
parser.add_argument('--baseline', type=int, default=0)
parser.add_argument('--calc_err', type=int, default=0)
parser.add_argument('--save_soi', type=int, default=0)
parser.add_argument('--pgd_param', default='0.125,0.007,7')


args = parser.parse_args()

if args.type == 'cifar10':
    print('loading cifar10 dataset')
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1, train= True)
    import cifar10_net
    import cifar10_net_infer

    # model = cifar10_net.Net().cuda()
    model = models.vgg16_bn() #cifar10(args=args, logger=logger)
    model.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=512, out_features=256, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=256, out_features=10, bias=True))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_adv_gen).state_dict())
    # model_infer = cifar10_net_infer.Net().cuda()
    # print('hello')
    model_infer = models.vgg16_bn() #model.cifar10(args=args, logger=logger)
    model_infer.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=512, out_features=256, bias=True),
                                   nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                   nn.Linear(in_features=256, out_features=10, bias=True))
    model_infer = torch.nn.DataParallel(model_infer)
    try:
        model_infer.load_state_dict(torch.load(args.model_inference).state_dict())
    except:
        model_infer.load_state_dict(torch.load(args.model_inference))

    if args.baseline == 1:
        try:
            model_infer.load_state_dict(torch.load(args.baseline_model).state_dict())
        except:
            model_infer.load_state_dict(torch.load(args.baseline_model))

    lut = torch.load(args.lut_path)
    bin_arr = lut[0]
    bin_arr = bin_arr[:300]   # Change this value to the number of bins in the LUT
    list_probs = lut[1]

    model = model.cuda()
    model_infer = model_infer.cuda()

elif args.type == 'cifar100':  #### change dataset to CIFAR100 if b present
    print('loading cifar100 dataset')
    # train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1, train=True)
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
        model_infer.load_state_dict(torch.load(args.model_inference).state_dict())
    except:
        model_infer.load_state_dict(torch.load(args.model_inference))
    model = model.cuda()
    model_infer = model_infer.cuda()
    if args.baseline == 1:
        try:
            model_infer.load_state_dict(torch.load(args.baseline_model).state_dict())
        except:
            model_infer.load_state_dict(torch.load(args.baseline_model))

    lut = torch.load(args.lut_path)
    bin_arr = lut[0]
    bin_arr = bin_arr[:300]
    list_probs = lut[1]

elif args.type == 'tinyimagenet':
    print('loading tinyimagenet dataset')
    train_loader, test_loader = dataset.tinyimagenet(batch_size=args.batch_size)
    import tinyimagenet_net
    import tinyimagenet_net_infer
    model = tinyimagenet_net.ResNet().cuda()
    model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load('./tinyimagenet/adv_train_pgd8_newdata/good_tinyimagenet.pth'))
    # model.load_state_dict(torch.load('./tinyimagenet/adv_train_pgd8/adv_train_pgd8.pth'))
    model.load_state_dict(torch.load('./tinyimagenet/baseline/tinyimagenet_baseline.pth'))
    model_infer = tinyimagenet_net_infer.ResNet().cuda()
    model_infer = torch.nn.DataParallel(model_infer)
    # model_infer.load_state_dict(torch.load('./tinyimagenet/train_pgd_16_8.pth'))
    # model.load_state_dict(torch.load('./tinyimagenet/adv_train_pgd8/adv_train_pgd8.pth'))

    model_infer.load_state_dict(torch.load('./tinyimagenet/adv_train_pgd8_newdata/good_tinyimagenet.pth'))
    if args.baseline == 1:
        model.load_state_dict(torch.load('./tinyimagenet/baseline/tinyimagenet_baseline.pth'))
        model_infer.load_state_dict(torch.load('./tinyimagenet/baseline/tinyimagenet_baseline.pth'))
    lut = torch.load('./tinyimagenet/LUT')
    bin_arr = lut[0]
    bin_arr = bin_arr[:300]
    list_probs = lut[1]


if args.a_type == 'fgsm' and args.clean == 0:
    eps,alpha,steps = map(float, args.pgd_param.split(','))
    print(f'FGSM with eps = {eps}')
    soi_path = args.soi_path + '/soi_' + args.type + '_' + args.a_type + '_e=' + str(eps)

if args.a_type == 'pgd' and args.clean == 0:
    eps,alpha,steps = map(float, args.pgd_param.split(','))
    steps = int(steps)
    print(f'PGD with eps = {eps}, alpha = {alpha}, steps = {steps}')
    soi_path = args.soi_path + '/soi_' + args.type + '_' + args.a_type+'_e=' + str(eps)+'_a=' + str(alpha)+'_n=' + str(steps)
if args.clean == 1:
    print(f'Using clean inputs')
    soi_path = args.soi_path + '/soi_' + args.type + '_clean'

count = 0
adv_data_batch = []
adv_label_batch = []
adv_test_set = []
soi_list = []
correct = 0
error = 0
accuracy = 0
access_list = []
model_infer.eval()
for i, (data, labels_tru) in enumerate(test_loader):

    indx_target = labels_tru.cpu().clone()
    if args.clean != 1:
        if args.a_type == 'pgd':
            inp_adv = adv_attacks.pgd_attack(model, data.cuda(), labels_tru.cuda(), eps, alpha, steps )
        elif args.a_type == 'fgsm':
            inp_adv = adv_attacks.fgsm_attack(data, eps, data_grad)
    else:
        inp_adv = data.cuda()
    output = model_infer(inp_adv)
    soi = model_infer.module.features[0](inp_adv).abs().mean(dim=1).mean(dim=1).mean(dim=1)

    if args.calc_err == 1:
        if args.baseline == 0:
            for idx, si in enumerate(soi):
                length = len(bin_arr[bin_arr<=si.detach().cpu().numpy()])
                prob_clean = list_probs[length-1]
                if bin_arr[len(bin_arr)-1] <= si:
                    prob_clean = 0

                rand_no = np.random.random_sample()
                if args.clean == 1 and rand_no <= prob_clean:
                    pred = torch.argmax(output[idx,:].data)
                    accuracy += pred.cpu().eq(indx_target[idx]).sum()
                if args.clean == 0 and rand_no <= prob_clean:

                    pred = torch.argmax(output[idx,:].data)
                    error += 1-pred.cpu().eq(indx_target[idx]).sum()

        else:
            if args.clean == 1:
                pred = output.data.max(1)[1]
                accuracy += pred.cpu().eq(indx_target).sum()
            elif args.clean == 0:
                pred = output.data.max(1)[1]
                error += (1-pred.cpu().eq(indx_target)).sum()

    tuple_da = (soi, labels_tru)

    soi_list.append(tuple_da)

print(f'accuracy: {accuracy}, error: {error}')
if args.save_soi == 1:
    print(f'saving soi at : {soi_path}')
    torch.save(soi_list, soi_path)
else:
    print('soi not saved')


