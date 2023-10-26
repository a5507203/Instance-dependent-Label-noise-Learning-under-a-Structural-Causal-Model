import math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from encoders import *
from PreResNet import *



class BaseVAE(nn.Module):
    def __init__(self, feature_dim=28, num_hidden_layers=1, hidden_size=32, z_dim =4, num_classes=100):
        super().__init__()
        self.z_encoder = Z_Encoder(feature_dim=feature_dim, num_classes=num_classes, num_hidden_layers=num_hidden_layers, hidden_size = hidden_size, z_dim=z_dim, lambda_dim = z_dim*(z_dim-1)//2)
        self.x_decoder = X_Decoder(feature_dim=feature_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, hidden_size = hidden_size, z_dim=z_dim)
        self.t_decoder = T_Decoder(feature_dim=feature_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, hidden_size = hidden_size)

    def _y_hat_reparameterize(self, c_logits):
        return F.gumbel_softmax(c_logits, dim=1)

    def _z_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std


    def forward(self, x: torch.Tensor, net): 
        c_logits = net(x)
        y_hat = self._y_hat_reparameterize(c_logits)
        mu, logvar = self.z_encoder(x, y_hat)
        z = self._z_reparameterize(mu, logvar)
        x_hat = self.x_decoder.forward(z=z, y_hat=y_hat)
        n_logits = self.t_decoder(x_hat, y_hat)

        return x_hat, n_logits, mu, logvar, c_logits, y_hat


class VAE_FASHIONMNIST(BaseVAE):
    def __init__(self, feature_dim=28, input_channel=1, z_dim =10, num_classes=10):
        super().__init__()
        self.z_encoder = CONV_Encoder_FMNIST(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_FMNIST(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder(feature_dim=feature_dim, in_channels =input_channel, num_classes=num_classes)


class VAE_CIFAR10(BaseVAE):
    def __init__(self, feature_dim=32, input_channel=3, z_dim =25, num_classes=10):
        super().__init__()
        self.z_encoder = CONV_Encoder_CIFAR(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CIFAR(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)

class VAE_CIFAR100(BaseVAE):
    def __init__(self, feature_dim=32, input_channel=3, z_dim =25, num_classes=100):
        super().__init__()
        self.z_encoder = CONV_Encoder_CIFAR(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CIFAR(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)
    

class VAE_CLOTHING1M(BaseVAE):
    def __init__(self, feature_dim=224, input_channel=3, z_dim =25, num_classes=10):
        super().__init__()
     
        self.z_encoder = CONV_Encoder_CLOTH1M(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CLOTH1M(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder_CLOTH1M(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)


def VAE_SVHN(feature_dim=32, input_channel=3, z_dim = 4, num_classes=10):
    return VAE_CIFAR10(feature_dim=feature_dim, input_channel=input_channel, z_dim =z_dim, num_classes=num_classes)


def VAE_MNIST( feature_dim=28, input_channel=1, z_dim = 4, num_classes=10):
    return VAE_FASHIONMNIST(feature_dim=feature_dim, input_channel=input_channel, z_dim =z_dim, num_classes=num_classes)

class VAE_WEBVISION(nn.Module):
    def __init__(self, feature_dim=299, num_hidden_layers=1, hidden_size=25, z_dim =100, num_classes=50, input_channel = 3):
        super().__init__()
        self.z_encoder = CONV_Encoder_WEBVISION(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_WEBVISION(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder_WEBVISION(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)
        self.kl_divergence = None
        self.flow  = None
    def _y_hat_reparameterize(self, c_logits):
        return F.gumbel_softmax(c_logits)

    def _z_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, x, net): 
        c_logits = net.forward(x)
        y_hat = self._y_hat_reparameterize(c_logits)
        mu, logvar = self.z_encoder(x, y_hat)
        z = self._z_reparameterize(mu, logvar)
        x_hat = self.x_decoder.forward(z=z, y_hat=y_hat)
        x_hat = torch.sigmoid(input=x_hat)
        n_logits = self.t_decoder(x_hat, y_hat)

        return x_hat, n_logits, mu, logvar, c_logits, y_hat