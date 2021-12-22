import torch
import torchvision
import torchvision.transforms as transforms
from .subset import Subset
import mylib
from mylib.data.data_loader.utils import create_train_val
from .dataloader import DataLoader_noise
import numpy as np
__all__ = ["load_noisydata"]

data_info_dict = {
    "CIFAR10":{
        "mean":(0.4914, 0.4822, 0.4465),
        "std":(0.2023, 0.1994, 0.2010),
        "root": "~/.torchvision/datasets/cifar10",
        'random_crop':32
    },
    "CIFAR100":{
        "mean":(0.4914, 0.4822, 0.4465),
        "std":(0.2023, 0.1994, 0.2010),
        "root": "~/.torchvision/datasets/cifar100",
        'random_crop':32
    },
    "SVHN":{
        "mean":(0.5, 0.5, 0.5),
        "std":(0.5, 0.5, 0.5),
        "root": "~/.torchvision/datasets/SVHN",
        'random_crop':32
    },
    "MNIST":{
        "mean":(0.1306604762738429,),
        "std":(0.30810780717887876,),
        "root": "~/.torchvision/datasets/MNIST",
        'random_crop':28
    },
    "FASHIONMNIST":{
        "mean":(0.286,),
        "std":(0.353,),
        "root": "~/.torchvision/datasets/FashionMNIST",
        'random_crop':28
    }
}
        


def load_noisydata(dataset = "CIFAR10", num_workers = 0, batch_size = 128, add_noise = False, noise_type = None, flip_rate_fixed = None, random_state = 1, trainval_split=None, train_frac = 1, augment = True):
  
    def transform_target(label):
        label = np.array(label)
        target = torch.from_numpy(label).long()
        return target  

    print('=> preparing data..')
 
    dataset = dataset.upper()
    info = data_info_dict[dataset]

    root = info["root"]
    random_crop = info["random_crop"]
    normalize = transforms.Normalize(info["mean"], info["std"])
    if augment != False:
        transform_train = transforms.Compose([transforms.RandomCrop(random_crop, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor() ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])



    test_dataset = mylib.data.dataset.__dict__[dataset+"_noise"](root=root, train=False, transform=transform_test, transform_eval=transform_test, add_noise = False, target_transform = transform_target)

    train_val_dataset = mylib.data.dataset.__dict__[dataset+"_noise"](
        root = root, 
        train = True, 
        transform = transform_train, 
        transform_eval = transform_test,
        target_transform = transform_target,
        add_noise = True,
        noise_type = noise_type, 
        flip_rate_fixed = flip_rate_fixed,
        random_state = random_state
    )

    train_dataset, val_dataset = create_train_val(train_val_dataset,trainval_split,train_frac)
    train_val_dataset = Subset(train_val_dataset, list(range(0, len(train_val_dataset), 1))) 
    train_val_loader = DataLoader_noise(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader_noise(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    est_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    test_loader = DataLoader_noise(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    

    return train_val_loader, train_loader, val_loader, est_loader, test_loader



