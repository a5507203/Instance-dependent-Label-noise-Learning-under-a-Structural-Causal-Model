#!/bin/bash


for dataset in CIFAR10 FASHIONMNIST CIFAR100 SVHN
do
    for seed in 1 2 3 
    do
        for z_dim in 25
        do

            for rate in 0.5 0.45 0.3 0.2
            do
                python3 main.py --flip_rate_fixed ${rate} --seed ${seed} --z_dim ${z_dim} --dataset ${dataset} > ${dataset}_flip_rate${rate}_seed${seed}_z_dim${z_dim}.out
            done

        done
    done
done

