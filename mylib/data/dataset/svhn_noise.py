from .torchvisiondataset import VisionDataset
from PIL import Image
import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple
from .torchvisiondatasetsutils import download_url, check_integrity, verify_str_arg
from .util import noisify
import torch

__all__ = ["SVHN_noise"]

class SVHN_noise(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            transform_eval : Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            add_noise: bool = True, 
            flip_rate_fixed: float = 0, 
            noise_type: str = '',      
            random_state = 1,        
            download: bool = True,
    ) -> None:
        super(SVHN_noise, self).__init__(root, transform=transform,
                                   target_transform=target_transform)

        if train == True:
            split = "train"
        else:
            split ="test"
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        self.t_matrix = None
        self.apply_transform_eval = False
        self.transform_eval = transform_eval     

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.clean_targets =self.targets.copy()
        self.is_confident = np.zeros(len(self.clean_targets))

        if add_noise:
            noisy_targets, self.actual_noise_rate, self.t_matrix = noisify(
                dataset=zip(torch.from_numpy(self.data.reshape(-1,3072)).float(),  torch.from_numpy(self.targets)),
                train_labels=self.targets, 
                noise_type=noise_type, 
                noise_rate=flip_rate_fixed, 
                random_state=random_state,
                nb_classes=self._get_num_classes(),
                feature_size=32*32*3
            )
            noisy_targets = noisy_targets.squeeze()
            self._set_targets(noisy_targets)

        self.hat_clean_targets = self.targets.copy()
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, clean_target, hat_clean_target, confidenice = self.data[index], int(self.targets[index]), int(self.clean_targets[index]), int(self.hat_clean_targets[index]), int(self.is_confident[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

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

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

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