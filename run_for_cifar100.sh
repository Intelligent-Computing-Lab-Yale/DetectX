#!/bin/bash

mkdir cifar100

python adv_train_set_gen.py --type=cifar100 --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path=./cifar100 --model_pth=./cifar100_baseline_vgg16_bn.pth

python adv_dataset_gen.py --type=cifar100 --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path=./cifar100 --model_pth=./cifar100_baseline_vgg16_bn.pth

python phase1_training.py --type=cifar100 --batch_size=200 --epochs=257 --la=0.6 --lc=0.1 --lr=0.01 --pgd_params=8_4_10 --baseline_model_pth=./cifar100_baseline_vgg16_bn.pth --train_pth=./cifar100/adv_train_set_cifar100_pgd_e=0.04_a=0.016_n=10 --test_pth=./cifar100/adv_test_set_cifar100_pgd_e=0.04_a=0.016_n=10 --save_pth=./cifar100

python phase2_training.py --type=cifar100 --batch_size=200 --path=./cifar100 --pgd_params=0.016,0.008,10 --adv_train=1 --net=vgg16_bn --model_adv_gen=./cifar100_baseline_vgg16_bn.pth --phase1_model_pth=./cifar100/cifar100_8_4_10.pth

python soi_lut_gen.py --type=cifar100 --a_type=pgd --lut_path=./cifar100 --model_adv_gen=./cifar100_baseline_vgg16_bn.pth --model_dual_phase=./cifar100/cifar100_vgg16_bn_adv_train_phase2.pth --pgd_param=0.04,0.016,10

python calculate_error_accuracy.py --type=cifar100 --a_type=pgd --pgd_param=0.04,0.016,10 --soi_path=./cifar100 --model_adv_gen=./cifar100_baseline_vgg16_bn.pth --model_inference=./cifar100/cifar100_vgg16_bn_adv_train_phase2.pth --lut_path=./cifar100/cifar100_pgd_LUT --clean=1 --baseline=0 --calc_err=1 --save_soi=1
