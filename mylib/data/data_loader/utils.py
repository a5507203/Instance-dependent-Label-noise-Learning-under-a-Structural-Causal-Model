import torch
from .subset import random_split, Subset
import numpy as np

def create_train_val(dataset, trainval_split, train_frac):
    total_len = len(dataset)
    print(total_len)
    if trainval_split:
        print("split validation set from training data")
        train_size = int(trainval_split * total_len)
        val_size = total_len - train_size
        train_size = int(train_frac*train_size)
        train_dataset, val_dataset,_ = random_split(dataset, [train_size, val_size, total_len-train_size-val_size])  
        
    else:
        print("use training data for validation")
        train_size = int(train_frac*total_len)
        train_dataset, _ = random_split(dataset, [train_size,total_len-train_size])  
        val_dataset = train_dataset
    if type(train_dataset.indices) == torch.Tensor:
        train_dataset.indices = train_dataset.indices.tolist()
    if type(val_dataset.indices) == torch.Tensor:
        val_dataset.indices = val_dataset.indices.tolist()

    return train_dataset, val_dataset


