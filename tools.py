import os
import numpy as np
import torch
from math import inf
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
from numpy.testing import assert_array_almost_equal

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed): 
    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 
    print("building dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1. - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)

# basic function
def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
#    print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    noise_label = []
    num_labels = len(y_train)
    idx = list(range(num_labels))
    random.shuffle(idx)
    num_noise = int(noise*num_labels)            
    noise_idx = idx[:num_noise]
    for i in range(num_labels):
        if i in noise_idx:
            noiselabel = random.randint(0,nb_classes-1)
            noise_label.append(noiselabel)
        else:    
            noise_label.append(y_train[i])   
    
    y_train_noisy = np.array(noise_label)
    y_train = y_train_noisy
    return y_train


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    y_train = np.array(y_train).reshape((-1,1))
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print (P)

    return y_train

def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1).unsqueeze(1)
    T_norm = row_abs / row_sum
    return T_norm



def fit(X, num_classes, percentage, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c)) # +1 -> index 
    eta_corr = X
    ind = []
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], percentage,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
            ind.append(idx_best)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
            
    return T, ind

def data_split(data, clean_labels, noisy_labels, split_percentage, seed=1):
   
    num_samples = int(clean_labels.shape[0])
    np.random.seed(int(seed))
    train_set_index = np.random.choice(num_samples, int(num_samples*split_percentage), replace=False)
    index = np.arange(data.shape[0])
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = data[train_set_index, :], data[val_set_index, :]
    train_clean_labels, val_clean_labels = clean_labels[train_set_index], clean_labels[val_set_index]
    train_noisy_labels, val_noisy_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_noisy_labels, val_noisy_labels,train_clean_labels, val_clean_labels,

def dataset_split_clothing1m(train_images, train_labels, split_per=0.9, random_seed=1):
    clean_train_labels = train_labels[:, np.newaxis]

    noisy_labels = clean_train_labels.squeeze()

    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels    

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-1)
            
    return net

