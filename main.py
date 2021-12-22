

import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from torchvision.utils import save_image
from torch.autograd import Variable
from mylib.utils import AverageMeter, ProgressMeter, fix_seed, accuracy, adjust_learning_rate, save_checkpoint
from mylib.data.data_loader import load_noisydata
from causalNL import run_vae
import numpy as np
# --- parsing and configuration --- #
parser = argparse.ArgumentParser(
    description="PyTorch implementation of VAE")
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--z_dim', type=int, default=25,
                    help='dimension of hidden variable Z (default: 10)')
parser.add_argument('--num_hidden_layers', type=int, default=2,
                    help='num hidden_ ayers (default: 0)')
parser.add_argument('--flip_rate_fixed', type=float,
                    help='fixed flip rates.', default=0.45)
parser.add_argument('--train_frac', default=1.0, type=float,
                    help='training sample size fraction')
parser.add_argument('--noise_type', type=str, default='instance')
parser.add_argument('--trainval_split',  default=1, type=float,
                    help='training set ratio')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--dataset', default="CIFAR10", type=str,
                    help='db')
parser.add_argument('--select_ratio', default=0, type=float,
                    help='confidence example selection ratio')
parser.add_argument('--pretrained', default=0, type=int,
                    help='using pretrained model or not')

arch_dict = {"FASHIONMNIST":"resnet18","CIFAR10":"resnet34","CIFAR100":"resnet50","SVHN":"resnet34"}
input_channel_dict = {"FASHIONMNIST":1,"CIFAR10":3,"CIFAR100":3,"SVHN":3}

def main():
    args = parser.parse_args()
    base_dir = "./"+args.dataset+"/"+args.noise_type+str(args.flip_rate_fixed)+"/"+str(args.seed)+"/"
    print(args)
    
    if args.seed is not None:
        fix_seed(args.seed)
    train_val_loader, train_loader, val_loader, est_loader, test_loader = load_noisydata(
        dataset = args.dataset,  
        noise_type = args.noise_type,
        random_state = args.seed, 
        batch_size = args.batch_size, 
        add_noise = True, 
        flip_rate_fixed = args.flip_rate_fixed, 
        trainval_split = args.trainval_split,
        train_frac = args.train_frac
    )
    test_dataset = test_loader.dataset
    train_dataset = train_loader.dataset


    print(train_loader.dataset._get_num_classes())
    

    out_dir = base_dir+"/co_teaching/"

    print("++++++++++++run co-teaching vae+++++++++++++++++")
    run_vae(
        train_loader = train_loader, 
        est_loader=est_loader, 
        test_loader= test_loader, 
        batch_size=args.batch_size, 
        epochs= args.epochs,
        z_dim= args.z_dim,
        cls_model = None,
        out_dir = out_dir +"vae/",
        select_ratio = args.select_ratio,
        pretrained = args.pretrained,
        dataset=args.dataset,
        noise_rate=args.flip_rate_fixed
    )



if __name__ == "__main__":
    main()