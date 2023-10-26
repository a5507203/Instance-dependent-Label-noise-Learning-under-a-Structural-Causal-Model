import torch
import numpy as np
import torchvision.transforms as transforms

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def transform_train(dataset_name):
    
    if dataset_name == 'mnist' or dataset_name == 'fashionmnist':
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),(0.3081, )),
       ])
    elif dataset_name == 'clothing1m':
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ])
    elif dataset_name == 'webvision':
        transform = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 
    else:
        transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
   ])
    
    return transform


def transform_test(dataset_name):
    
    if dataset_name == 'mnist' or dataset_name == 'fashionmnist':
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),(0.3081, )),
       ])
    elif dataset_name == 'clothing1m':
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])
    elif dataset_name == 'webvision':
        transform = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])  
    else:
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
   ])
    
    return transform


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    
