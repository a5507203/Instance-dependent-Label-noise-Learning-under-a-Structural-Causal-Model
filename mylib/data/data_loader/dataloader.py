
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np

__all__ = ["DataLoader_noise"]

class DataLoader_noise(DataLoader):

    def eval(self):
        self.dataset.eval()

    def train(self):
        self.dataset.train()