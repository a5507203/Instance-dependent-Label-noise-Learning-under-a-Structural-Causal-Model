


import argparse
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from PIL import Image
from torchvision.utils import save_image
from torch.autograd import Variable
from mylib.utils import AverageMeter, ProgressMeter, fix_seed, accuracy, adjust_learning_rate, save_checkpoint
from causalNL import run_vae

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data as Data

# --- parsing and configuration --- #
parser = argparse.ArgumentParser(
    description="PyTorch implementation of VAE")
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--z_dim', type=int, default=50,
                    help='dimension of hidden variable Z (default: 10)')
parser.add_argument('--num_hidden_layers', type=int, default=2,
                    help='num hidden_ ayers (default: 0)')
parser.add_argument('--flip_rate_fixed', type=float,
                    help='fixed flip rates.', default=0.4)
parser.add_argument('--train_frac', default=1.0, type=float,
                    help='training sample size fraction')
parser.add_argument('--noise_type', type=str, default='instance')
parser.add_argument('--trainval_split',  default=1, type=float,
                    help='training set ratio')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--dataset', default="CLOTH1M", type=str,
                    help='db')
parser.add_argument('--select_ratio', default=0, type=float,
                    help='confidence example selection ratio')
parser.add_argument('--pretrained', default=0, type=int,
                    help='using pretrained model or not')

## hardcode
arch_dict = {"CLOTH1M":"resnet50"}
input_channel_dict = {"CLOTH1M":3}



def dataset_split_clothing(train_images, train_labels, split_per=1, random_seed=1, num_classes=14):
    total_labels = train_labels[:, np.newaxis]
    num_samples = int(total_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)
    print(train_images.shape)
    train_set, val_set = train_images[train_set_index], train_images[val_set_index]
    train_labels, val_labels = total_labels[train_set_index], total_labels[val_set_index]
    return train_set, val_set, train_labels, val_labels 


class clothing1m_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9, random_seed=1, num_class=14):
        self.imgs = np.load('images/index/noisy_train_images.npy')
        self.labels = np.load('images/index/noisy_train_labels.npy')
        self.train_data, self.val_data, self.train_labels, self.val_labels = dataset_split_clothing(self.imgs,
                                                                                                          self.labels,
                                                                                                          1,
                                                                                                          random_seed,
                                                                                                          num_class)
        self.train_labels = self.train_labels.squeeze()
        self.val_labels = self.val_labels.squeeze()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        if self.train:
            fn = self.train_data[index]
            img = Image.open(fn).convert('RGB')
            label = self.train_labels[index]
        else:
            fn = self.val_data[index]
            img = Image.open(fn).convert('RGB')
            label = self.val_labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, label, None, None

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.val_labels)


class clothing1m_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.imgs = np.load('images/index/clean_test_images.npy')
        self.labels = np.load('images/index/clean_test_labels.npy')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, label, None, None

    def __len__(self):
        return len(self.imgs)

## TODO 1. replace datasets to cloth1M; 2. replace flip_rate_fixed to te noise rate of clothing1M

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target  

def Cloth1M_loader(num_workers = 0, batch_size = 128, trainval_split=None, train_frac=1):
    print('=> preparing data..')
    
    transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])  

    test_dataset = clothing1m_test_dataset(transform=transform_test,  target_transform = transform_target)
    train_dataset = clothing1m_dataset( transform=transform_train,  target_transform = transform_target)
  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    return train_loader, test_loader, train_dataset,  test_dataset



def main():
    args = parser.parse_args()
    base_dir = "./"+args.dataset+"/"+args.noise_type+str(args.flip_rate_fixed)+"/"+str(args.seed)+"/"
    print(args)
    
    train_loader, test_loader, _,  _ = Cloth1M_loader(batch_size=args.batch_size)

    

    out_dir = base_dir+"/co_teaching/"

    print("++++++++++++run co-teaching vae+++++++++++++++++")
    run_vae(
        train_loader = train_loader, 
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