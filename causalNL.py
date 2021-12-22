
from math import e
import os
from mylib.models import vae
from mylib.data.data_loader.dataloader import DataLoader_noise
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.datasets.mnist import FashionMNIST
from torchvision.utils import save_image
from torch.autograd import Variable
from mylib import models
from mylib.utils import AverageMeter, ProgressMeter, fix_seed, accuracy, save_checkpoint
import types
import numpy as np
# --- parsing and configuration --- #
from collections import OrderedDict, defaultdict
import mylib.models as models
from mylib.utils import save_model


vae_args = types.SimpleNamespace()

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=vae_args.alpha_plan[epoch]
        param_group['betas']=(vae_args.beta1_plan[epoch], 0.999) # Only change beta1
        
        

def log_standard_categorical(p, reduction="mean"):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    # print(cross_entropy)
  
    if reduction=="mean":
        cross_entropy = torch.mean(cross_entropy)
    else:
        cross_entropy = torch.sum(cross_entropy)
    
    return cross_entropy



def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember



def vae_loss(x_hat, data, n_logits, targets, mu, log_var, c_logits, h_c_label):
    # x loss 
    l1 = 0.1*F.mse_loss(x_hat, data, reduction="mean")

    # \tilde{y]} loss
    l2 = 0.1*F.cross_entropy(n_logits, targets, reduction="mean")
    #  uniform loss for x
    l3 = -0.00001*log_standard_categorical(h_c_label, reduction="mean")
    #  Gaussian loss for z
    l4 = - 0.01 *torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (l1+l2+l3+l4)

# --- train and test --- #
def train(epoch, model, train_loader, optimizers, device):

    n_top1 = AverageMeter('Acc@1', ':6.2f')
    co1_loss = AverageMeter('Acc@1', ':6.2f')
    co2_loss = AverageMeter('Acc@1', ':6.2f')
    vae1_loss = AverageMeter('Acc@1', ':6.2f')
    vae2_loss = AverageMeter('Acc@1', ':6.2f')
    vae_model1 = model["vae_model1"].train()
    vae_model2 = model["vae_model2"].train()
    optimizer1 = optimizers["vae1"]
    optimizer2 = optimizers["vae2"]

    for _, (data, targets, _, _, _) in enumerate(train_loader):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
     
        #forward
        x_hat1, n_logits1, mu1, log_var1, c_logits1, y_hat1  = vae_model1(data)
        x_hat2, n_logits2, mu2, log_var2, c_logits2, y_hat2 = vae_model2(data)
        #calculate acc
        n_acc1, _ = accuracy(n_logits1, targets, topk=(1, 2))

        n_top1.update(n_acc1.item(), data.size(0))

 
        # calculate loss
        vae_loss_1 = vae_loss(x_hat1, data, n_logits1, targets, mu1, log_var1, c_logits1, y_hat1)
        vae_loss_2 = vae_loss(x_hat2, data, n_logits2, targets, mu2, log_var2, c_logits2, y_hat2)

        co_loss_1, co_loss_2 = loss_coteaching(c_logits1, c_logits2, targets, vae_args.rate_schedule[epoch])

        loss_1 =  co_loss_1+vae_loss_1
        loss_2 =   co_loss_2+vae_loss_2
        co1_loss.update(co_loss_1.item(), data.size(0))
        co2_loss.update(co_loss_2.item(), data.size(0))
        vae1_loss.update(vae_loss_1.item(), data.size(0))
        vae2_loss.update(vae_loss_2.item(), data.size(0))
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

    print('====> Epoch: {} Average loss: {:.5f}/{:.5f}/{:.5f}/{:.5f}'.format( epoch,co1_loss.avg, co2_loss.avg,vae1_loss.avg,vae2_loss.avg))
    print('====> train noisy acc: {:.4f}'.format(n_top1.avg))

  



def test(epoch, model, test_loader, device, dataset):
    top1 = AverageMeter('Acc@1', ':6.2f')
    vae_model1 = model["vae_model1"].eval()
    vae_model1 = model["vae_model1"].eval()
    new_labels  = []
    recon_points = []
    with torch.no_grad():
        for batch_idx, (data, _, clean_targets, _, _)  in enumerate(test_loader):
            data = data.to(device)
            clean_targets = clean_targets.to(device)
            x_hat, _, _, _, c_logits,_ = vae_model1(data)
 
            # calculate the training acc
            h_c_acc1, _ = accuracy(c_logits, clean_targets, topk=(1, 2))
            top1.update(h_c_acc1.item(), data.size(0))
    
            max_probs, target_u = torch.max(c_logits, dim=-1)
            recon_points += x_hat.tolist()
            new_labels +=target_u.tolist()

    print('====> Test1 set acc: {:.4f}'.format(top1.avg))
    return top1.avg,  top1.avg




# --- etc. funtions --- #
def save_generated_img(image, name, epoch, nrow=8):
    if not os.path.exists('results'):
        os.makedirs('results')
    if epoch % 5 == 0:
        save_path = 'results/'+name+'_'+str(epoch)+'.png'
        save_image(image, save_path, nrow=nrow)



def run_vae(
    train_loader, 
    test_loader, 
    batch_size=128, 
    epochs=100, 
    z_dim=2, 
    est_loader= None,
    cls_model = None, 
    out_dir = "", 
    select_ratio =0.25, 
    pretrained = 0, 
    dataset="CIFAR10",
    noise_rate = 0.45
    ):
    vae_args.lr = 0.001
    vae_args.LOG_INTERVAL = 100
    vae_args.BATCH_SIZE = batch_size
    vae_args.EPOCHS = epochs
    vae_args.z_dim = z_dim
    vae_args.pretrained = pretrained
    vae_args.dataset = dataset
    vae_args.select_ratio = select_ratio
    vae_args.epoch_decay_start = 1000
    vae_args.noise_rate = noise_rate
    vae_args.forget_rate = noise_rate
    vae_args.exponent = 1
    vae_args.num_gradual = 10
    mom1 = 0.9
    mom2 = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(vae_args)
    best_acc = 0
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    vae_args.alpha_plan = [vae_args.lr] * vae_args.EPOCHS
    vae_args.beta1_plan = [mom1] * vae_args.EPOCHS

    for i in range(vae_args.epoch_decay_start, vae_args.EPOCHS):
        vae_args.alpha_plan[i] = float(vae_args.EPOCHS - i) / (vae_args.EPOCHS - vae_args.epoch_decay_start) * vae_args.lr
        vae_args.beta1_plan[i] = mom2

    vae_args.rate_schedule = np.ones(vae_args.EPOCHS)*vae_args.forget_rate 
    vae_args.rate_schedule[:vae_args.num_gradual] = np.linspace(0, vae_args.forget_rate **vae_args.exponent, vae_args.num_gradual)
    print(  vae_args.rate_schedule)
    # exit()

    if dataset == "CLOTH1M":
        vae_model1 = models.__dict__["VAE_"+vae_args.dataset](z_dim=vae_args.z_dim, num_classes=14)
        vae_model2 = models.__dict__["VAE_"+vae_args.dataset](z_dim=vae_args.z_dim, num_classes=14)
    else:
        vae_model1 = models.__dict__["VAE_"+vae_args.dataset](z_dim=vae_args.z_dim, num_classes=train_loader.dataset._get_num_classes())
        vae_model2 = models.__dict__["VAE_"+vae_args.dataset](z_dim=vae_args.z_dim, num_classes=train_loader.dataset._get_num_classes())

    model = {"vae_model1":vae_model1.to(device), "vae_model2":vae_model2.to(device)}

    optimizers = {'vae1':torch.optim.Adam(model["vae_model1"].parameters(), lr=vae_args.lr),'vae2':torch.optim.Adam(model["vae_model2"].parameters(), lr=vae_args.lr)}
    test_acc = 0
    for epoch in range(0, vae_args.EPOCHS):
        adjust_learning_rate(optimizers['vae1'], epoch)
       
        adjust_learning_rate(optimizers['vae2'], epoch)
        train(epoch, model, train_loader, optimizers, device)


        curr_test_acc1, curr_test_acc2 = test(epoch, model, test_loader, device, vae_args.dataset)
        # if vae_args.EPOCHS% 20 == 0:

        if vae_args.EPOCHS-epoch<=10:
            print(epoch)
            test_acc += curr_test_acc1

    test_acc = test_acc/10
    save_model(state ={  'epoch': epoch + 1,'state_dict': vae_model1.state_dict(), 'avg_acc1': test_acc, 'last_acc1':curr_test_acc1 },out=out_dir)
    print("vae avg acc1: ",test_acc)
    print("vae last acc1: ",curr_test_acc1) 
    return test_acc, model