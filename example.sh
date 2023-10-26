#!/bin/bash
 

python main.py --dataset fashionmnist --noise_type instance --noise_rate 0.2


python main.py --dataset cifar10 --noise_type instance --noise_rate 0.2
python main.py --dataset cifar100 --noise_type instance --noise_rate 0.4 
python main.py --dataset cifar10 --noise_type worse_label --noise_rate 0.4
python3 main.py --dataset cifar10 --noise_type random_label3 --noise_rate 0.2
python3 main.py --dataset cifar10 --noise_type random_label2 --noise_rate 0.2 
python3 main.py --dataset cifar10 --noise_type random_label1 --noise_rate 0.2
python3 main.py --dataset cifar10 --noise_type aggre_label --noise_rate 0.1