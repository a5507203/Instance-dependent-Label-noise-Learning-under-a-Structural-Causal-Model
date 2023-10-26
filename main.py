# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from model import CNN
import argparse, sys
import numpy as np
import datetime
import data_load
from transformer import transform_train, transform_test,transform_target
from itertools import chain
from PreResNet import ResNet18
from vae import VAE_CIFAR10, VAE_CIFAR100, VAE_FASHIONMNIST
import torchvision.transforms as transforms

from loss import loss_coteaching

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric, instance]', default='pairflip')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'fashionmnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=40)
parser.add_argument('--lambda_elbo', default=0.001, type=float, help='weight for elbo')

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 

# load dataset
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 20
    args.n_epoch = 40
    train_dataset = data_load.mnist_dataset(True,
                                        transform = transform_train(args.dataset), 
                                        target_transform=transform_target,
                                        noise_rate=args.noise_rate,
                                        noise_type=args.noise_type,
                                        split_percentage=1.0,
                                        seed=args.seed)

    test_dataset =  data_load.mnist_test_dataset(
                                        transform = transform_test(args.dataset),
                                        target_transform=transform_target)
    
# load dataset
if args.dataset=='fashionmnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 20
    args.n_epoch = 40
    train_dataset = data_load.fashionmnist_dataset(True,
                                        transform = transform_train(args.dataset), 
                                        target_transform=transform_target,
                                        noise_rate=args.noise_rate,
                                        noise_type=args.noise_type,
                                        split_percentage=1.0,
                                        seed=args.seed)

    test_dataset =  data_load.fashionmnist_test_dataset(
                                        transform = transform_test(args.dataset),
                                        target_transform=transform_target)

if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 40
    args.n_epoch = 100
    train_dataset = data_load.cifar10_dataset(True,
                                        transform = transform_train(args.dataset), 
                                        target_transform=transform_target,
                                        noise_rate=args.noise_rate,
                                        noise_type=args.noise_type,
                                        split_percentage=1.0,
                                        seed=args.seed)

    test_dataset =  data_load.cifar10_test_dataset(
                                        transform = transform_test(args.dataset),
                                        target_transform=transform_target)

if args.dataset=='cifar100':
    input_channel=3
    num_classes=100
    args.top_bn = False
    args.epoch_decay_start = 50
    args.n_epoch = 100
    train_dataset = data_load.cifar100_dataset(True,
                                        transform = transform_train(args.dataset), 
                                        target_transform=transform_target,
                                        noise_rate=args.noise_rate,
                                        noise_type=args.noise_type,
                                        split_percentage=1.0,
                                        seed=args.seed)

    test_dataset =  data_load.cifar100_test_dataset(
                                        transform = transform_test(args.dataset),
                                        target_transform=transform_target)

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
   
save_dir = args.result_dir +'/' +args.dataset+'/coteaching/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(args.noise_rate)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def my_vae_loss(x_hat, x, n_logits, targets, y_hat, mean, logvar):
    # c_bernoulli = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=x_hat)
    # recons_loss = torch.mean(- c_bernoulli.log_prob(value=x)) # (N, C, H, W)
    recons_loss = F.mse_loss(x_hat, x, reduction="mean")
    ce_loss = F.cross_entropy(n_logits, targets, reduction="mean")
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim = 1), dim = 0)
    entropy = y_hat*torch.log(y_hat+1e-8)
    entropy_loss = entropy.sum(1).mean(0)
    return recons_loss+ce_loss+kld_loss+entropy_loss, recons_loss, ce_loss, kld_loss, entropy_loss

# Train the Model
def train(train_loader,epoch, model1, vae1, optimizer1, model2, vae2, optimizer2):
    print('Training %s...' % model_str)
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    recall_ratio_1_list=[]
    recall_ratio_2_list=[]
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits1=model1(images)
        x_hat, n_logits, mu, logvar, c_logits, y_hat = vae1(images, model1)
        vae_loss1, recons_loss, ce_loss, kld_loss, entropy_loss = my_vae_loss(x_hat, images, n_logits, labels, y_hat, mu, logvar)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(images)
        x_hat, n_logits, mu, logvar, c_logits, y_hat = vae2(images, model2)
        vae_loss2, recons_loss, ce_loss, kld_loss, entropy_loss = my_vae_loss(x_hat, images, n_logits, labels, y_hat, mu, logvar)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2+=1
        train_correct2+=prec2
        loss_1, loss_2, pure_ratio_1, pure_ratio_2, recall_ratio_1, recall_ratio_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        total_loss1=loss_1+args.lambda_elbo*vae_loss1
        total_loss2=loss_2+args.lambda_elbo*vae_loss2
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)
        recall_ratio_1_list.append(100*recall_ratio_1)
        recall_ratio_2_list.append(100*recall_ratio_2)

        optimizer1.zero_grad()
        total_loss1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        total_loss2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f, Recall Ratio1: %.4f, Recall Ratio2 %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.item(), loss_2.item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list), np.sum(recall_ratio_1_list)/len(recall_ratio_1_list), np.sum(recall_ratio_2_list)/len(recall_ratio_2_list)))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, recall_ratio_1_list, recall_ratio_2_list

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2


def create_model(args, num_classes):
    if args.dataset == 'fashionmnist':
        model = ResNet18(num_classes=num_classes, in_c=1)
    else:
        model = ResNet18(num_classes=num_classes, in_c=3)
    if args.dataset=='cifar10' or args.dataset=='svhn':
        vae_model = VAE_CIFAR10(num_classes=num_classes)
    elif args.dataset=='fashionmnist':
        vae_model = VAE_FASHIONMNIST(num_classes=num_classes)
    elif args.dataset=='cifar100':
        vae_model = VAE_CIFAR100(num_classes=num_classes)
    
    total_params1 = sum(p.numel() for p in model.parameters())
    total_params2 = sum(p.numel() for p in vae_model.parameters())
    print(f"Number of parameters: {total_params1+total_params2}")
    model = model.cuda()
    vae_model = vae_model.cuda()
    return model, vae_model

def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')
    cnn1, vae1 = create_model(args, num_classes)
    optimizer1 = torch.optim.Adam(chain(cnn1.parameters(), vae1.parameters()), lr=learning_rate)
    
    cnn2, vae2 = create_model(args, num_classes)
    optimizer2 = torch.optim.Adam(chain(cnn2.parameters(), vae2.parameters()), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0
    mean_recall_ratio1=0
    mean_recall_ratio2=0

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%, Recall Ratio 1 %.4f %%, Recall Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2, mean_recall_ratio1, mean_recall_ratio2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, recall_ratio_1_list, recall_ratio_2_list=train(train_loader, epoch, cnn1, vae1, optimizer1, cnn2, vae2, optimizer2)
        # evaluate models
        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        mean_recall_ratio1 = sum(recall_ratio_1_list)/len(recall_ratio_1_list)
        mean_recall_ratio2 = sum(recall_ratio_2_list)/len(recall_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%, Recall Ratio 1 %.4f %%, Recall Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2, mean_recall_ratio1, mean_recall_ratio2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")

if __name__=='__main__':
    main()
