import collections
import glob
import os

import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from fastai.text import *
from fastai.lm_rnn import *

def train_loader(path, batch_size, num_workers=4, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    PRE = 'fwd_'
    IDS = 'ids'
    train_file_id = ''
    dir_path = Path(path)
    trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}{train_file_id}.npy')
    trn_lbls = np.load(dir_path / 'tmp' / f'lbl_trn{train_file_id}.npy')
    trn_lbls = trn_lbls.flatten()
    trn_lbls -= trn_lbls.min()
    trn_ds = TextDataset(trn_sent, trn_lbls)
    trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=batch_size//2)
    trn_dl = DataLoader(trn_ds, batch_size//2, transpose=True, num_workers=num_workers, pad_idx=1, sampler=trn_samp)
    return trn_dl

#    return data.DataLoader(trn_ds,
#        batch_size=batch_size,
#        shuffle=True,
#        num_workers=num_workers,
#        pin_memory=pin_memory)


def test_loader(path, batch_size, num_workers=4, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    PRE = 'fwd_'
    IDS = 'ids'
    train_file_id = ''
    dir_path = Path(path)
    val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')
    val_lbls = np.load(dir_path / 'tmp' / f'lbl_val.npy')
    val_lbls = val_lbls.flatten()
    val_lbls -= val_lbls.min()
    val_ds = TextDataset(val_sent, val_lbls)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    val_dl = DataLoader(val_ds, batch_size, transpose=True, num_workers=num_workers, pad_idx=1, sampler=val_samp)
    return val_dl
#   return data.DataLoader(val_ds,
#        batch_size=batch_size,
#        shuffle=False,
#        num_workers=num_workers,
#        pin_memory=pin_memory)


def test_loader_caffe(path, batch_size, num_workers=4, pin_memory=False):
    """Legacy loader for caffe. Used with models loaded from caffe."""
    # Returns images in 256 x 256 to subtract given mean of same size.
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((256, 256)),
                                 # transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


def train_loader_cropped(path, batch_size, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((224, 224)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


def test_loader_cropped(path, batch_size, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


# Note: This might not be needed anymore given that this functionality exists in
# the newer PyTorch versions.
class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)
