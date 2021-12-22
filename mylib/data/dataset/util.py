from __future__ import print_function
import os
import os.path
import copy
import hashlib
import errno
import numpy as np

import torch
from math import inf
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn



from numpy.testing import assert_array_almost_equal
def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
   
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
     
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
   

    return y_train, actual_noise, P

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, actual_noise, P



def check_no_extreme_noise(t):
    for i in range(len(t)):
        if np.argmax(t[i]) != i:
            # print(t)
            # exit()
            return False 
       
    return True

def CCN_generator_random(y_train, flip_rate_high, random_state=None, nb_classes = 10):
    P = []
    flipper = np.random.RandomState(random_state)
    for i in range(nb_classes):
        flip_rate = flipper.uniform(0,flip_rate_high, 1)[0]
        max_flip_rate = 1-flip_rate
        while True:
            avail_flip_rates = flip_rate
            row_flip_rates = []
            for _ in range(nb_classes-1):
                if avail_flip_rates>0:
                    curr_flip_rate = flipper.uniform(0,max_flip_rate, 1)[0]
                    if avail_flip_rates-curr_flip_rate<0:
                        curr_flip_rate = avail_flip_rates
                    avail_flip_rates -= curr_flip_rate
                    row_flip_rates.append(curr_flip_rate)
                else:
                    row_flip_rates.append(0.0)
            if (1-sum(row_flip_rates)) - max_flip_rate <0.0000001:
                break  
        flipper.shuffle(row_flip_rates)
        row_flip_rates.insert(i,1-sum(row_flip_rates))
        P.append(row_flip_rates)
    P = np.array(P)
    y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    return y_train_noisy, actual_noise, P


def CCN_generator_multiflip(y_train, flip_rate, random_state=None, nb_classes = 10):
    flipper = np.random.RandomState(random_state)
    P = []
    max_flip_rate = 1-flip_rate
    for i in range(nb_classes):
        while True:
            avail_flip_rates = flip_rate
            row_flip_rates = []
            for _ in range(nb_classes-1):
                if avail_flip_rates>0:
                    curr_flip_rate = flipper.uniform(0,max_flip_rate, 1)[0]
                    if avail_flip_rates-curr_flip_rate<0:
                        curr_flip_rate = avail_flip_rates
                    avail_flip_rates -= curr_flip_rate
                    row_flip_rates.append(curr_flip_rate)
                else:
                    row_flip_rates.append(0.0)
            if (1-sum(row_flip_rates)) - max_flip_rate <0.0000001:
                break  
        flipper.shuffle(row_flip_rates)
        row_flip_rates.insert(i,1-sum(row_flip_rates))
        P.append(row_flip_rates)
    P = np.array(P)
    y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    return y_train_noisy, actual_noise, P



def get_instance_noisy_label(n, dataset, labels, nb_classes, feature_size, norm_std, random_state): 
    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 
    print("building dataset...")
    label_num = nb_classes
    np.random.seed(int(random_state))
    torch.manual_seed(int(random_state))
    torch.cuda.manual_seed(int(random_state))
    print(dataset)
    print(labels)
    print(norm_std)
    print(random_state)
    P = []
    if n == 0:

        flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=0)
    else:
        flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
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
    return np.array(new_label), n, P


def noisify(dataset=None, nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0, feature_size = 28*28, norm_std=0.1):
    if noise_type == 'pairflip':
        train_labels = train_labels[:, np.newaxis]
        train_noisy_labels, actual_noise_rate, P = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    elif noise_type == 'symmetric':
        train_labels = train_labels[:, np.newaxis]
        train_noisy_labels, actual_noise_rate, P = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    elif noise_type == 'multiflip':
        train_labels = train_labels[:, np.newaxis]
        train_noisy_labels, actual_noise_rate, P = CCN_generator_multiflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes) 
        train_labels = train_labels[:, np.newaxis]
    elif noise_type == 'random':
        train_noisy_labels, actual_noise_rate, P = CCN_generator_random(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes) 
    elif noise_type == 'instance':
        train_labels = torch.from_numpy(train_labels)
        train_noisy_labels, actual_noise_rate, P = get_instance_noisy_label(n=noise_rate, dataset=dataset, labels=train_labels, nb_classes=nb_classes, feature_size=feature_size, norm_std=norm_std, random_state=random_state)

    else:
        print("invalid noise type")
        exit()
    return train_noisy_labels, actual_noise_rate, P