from torch.nn import Module
from torch import nn
import torch.nn.functional as F

__all__ = ["NaiveNet"]


class NaiveNet(nn.Module):

    def __init__(self, feature_dim=2, hidden_dim=5, num_classes=2, pretrained=False, input_channel=1):
        super(NaiveNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out=F.dropout(out, p=0.25)
        out = F.relu(self.fc2(out))
        # out=F.dropout(out, p=0.25)
        out = self.fc3(out)
        return out

