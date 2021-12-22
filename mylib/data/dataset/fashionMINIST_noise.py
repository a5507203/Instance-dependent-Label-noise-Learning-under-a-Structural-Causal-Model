from __future__ import print_function
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
from .torchvisiondataset import VisionDataset
from .torchvisiondatasetsutils import download_url, download_and_extract_archive, extract_archive, makedir_exist_ok, verify_str_arg
from .util import noisify

__all__ = ["FASHIONMNIST_noise", "MNIST_noise"]

class MNIST_noise(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self, 
        root, 
        train=True, 
        transform=None, 
        transform_eval = None,
        target_transform = None, 
        add_noise= True, 
        flip_rate_fixed = None, 
        noise_type = '', 
        random_state = 1, 
        download=True):
        super(MNIST_noise, self).__init__(root, transform=transform,
                                    target_transform=target_transform)


        self.t_matrix = None
        self.apply_transform_eval = False
        self.use_train = train  # training set or test set
        self.transform_eval = transform_eval     

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.use_train:
            data_file = self.training_file
        else:
            data_file = self.test_file


        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        #print(self.data.shape)
        # self.data = self.data.cpu().detach().numpy()
        self.data = self.data.numpy().reshape((-1, 28*28))
        self.targets = self.targets.cpu().detach().numpy()
        self.clean_targets = self.targets.copy()

        self.is_confident = np.zeros(len(self.clean_targets))
        
        if add_noise:
            noisy_targets, self.actual_noise_rate, self.t_matrix = noisify(
                dataset=zip(torch.from_numpy(self.data).float(),  torch.from_numpy(self.targets)),
                train_labels=self.targets, 
                noise_type=noise_type, 
                noise_rate=flip_rate_fixed, 
                random_state=random_state,
                nb_classes=self._get_num_classes(),
                feature_size=784
            )
            noisy_targets = noisy_targets.squeeze()
 
            self._set_targets(noisy_targets)
        self.hat_clean_targets = self.targets.copy()

        self.data = self.data.reshape((-1, 28,28))
        self.data = self.data.transpose((0, 2, 1)) 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, clean_target, hat_clean_target, confidenice = self.data[index], int(self.targets[index]), int(self.clean_targets[index]), int(self.hat_clean_targets[index]), int(self.is_confident[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.apply_transform_eval:
            transform = self.transform_eval
        else:
            transform = self.transform  

        if self.transform is not None:
            img = transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            clean_target = self.target_transform(clean_target)
            hat_clean_target = self.target_transform(hat_clean_target)
            # confidenice = self.target_transform(confidenice)
            
        return img, target, clean_target, hat_clean_target, confidenice

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))
    
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        if self._check_exists():
            return
        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

        
    def _set_targets(self,n_targets):
        self.targets = n_targets
 

    def _get_num_classes(self):
        return len(set(self.targets))

    def _get_targets(self):
        return self.targets.data.tolist()


    def train(self):
        self.apply_transform_eval = False


    def eval(self):
        self.apply_transform_eval = True

    def get_clean_ratio(self):
        correct = 0
        t_number = 0
        for (c_label, h_c_label, confidence) in zip(self.clean_targets, self.hat_clean_targets,  self.is_confident):
            if confidence == 1:
                if c_label == h_c_label:
                    correct +=1
                t_number +=1
        return correct/(t_number+1e-10)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.use_train is True else "Test")


class FASHIONMNIST_noise(MNIST_noise):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Fashion-MNIST/processed/training.pt``
            and  ``Fashion-MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    resources = [
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
         "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
         "25c81989df183df01b3e8a0aad5dffbe"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554ea6380940ec79"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a12a480ee83cd310")
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']







def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x
