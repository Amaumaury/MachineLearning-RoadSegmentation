import matplotlib.image as mpimg

import numpy as np

import utils

import torch
import torch.nn as nn

from sklearn.cluster import MeanShift

def load_image(filename):
    """Loads an image into a numpy array"""
    return mpimg.imread(filename)


def image_to_features(img, kernel_size, pad):
    """Linearizes patches of an image into lines.
    Arguments:
     :img: the image to linearize of shape (W, H, C)
     :kernel_size: the length of the side of the patch which will be squared
     should be odd.

    The radius of the patch is r = (kernel_size - 1) / 2
    The produced matrix has shape ((W * H, kernel_size**2 * C)
    """
    v = (kernel_size - 1) // 2
    if pad:
        img = reflect_padding(img, v)
    features = []
    for i in range(img.shape[0] - (kernel_size - 1)):
        for j in range(img.shape[1] - (kernel_size - 1)):
            newline = img[i : i + kernel_size, j : j + kernel_size]
            features.append(newline)
    return np.stack(features)


def reassemble(lines, kernel_size, original_w, original_h, channels=3):
    """Reassembles the matrix produced by image_to_features into an image"""
    v = (kernel_size - 1) // 2
    center_flat = v * kernel_size + v
    shifter = np.arange(lines.shape[0]) * lines.shape[1]
    shifter = np.reshape(shifter, (len(shifter), 1))
    shifter = np.tile(shifter, (1, channels))
    mask = np.array([center_flat * channels + i for i in range(channels)])
    mask = np.tile(mask, (lines.shape[0], 1))
    mask = shifter + mask
    mask = np.ravel(mask)
    flat = np.ravel(lines)
    pixels = flat[mask]
    return np.reshape(pixels, (original_w, original_h, channels))


def crop_groundtruth(img, kernel_size=None):
    #radius = (kernel_size - 1) // 2
    img[img < 0.5] = -1
    img[img >= 0.5] = 1
    #return img[radius : -radius, radius : -radius] not needed since image_to_features does padding
    return img


def preds_to_tensor(preds, kernel_size, n, w, h):
    """Transforms the flat vector of probability predicted into an image"""
    #return np.reshape(preds, (n, w - (kernel_size - 1), h - (kernel_size - 1))) not needed <= padding
    return np.reshape(preds, (n, w, h))


def patch_map(img, patch_size, f=lambda p: p):
    """Downsamples the provided image by moving a squared patch of size
    patch_size over it and applying, for each position, the function p
    """
    #assert all([dim % patch_size == 0 for dim in img.shape[:2]]), 'Dimensions are not divisible by patch size'
    rows = []
    for i in range(0, img.shape[0], patch_size):
        row = []
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            row.append(f(patch))
        rows.append(row)
    return np.array(rows)


def patch_iterator(img, patch_size, stride=1):
    assert all([dim % patch_size == 0 for dim in img.shape[:2]]), 'Dimensions are not divisible by patch size'
    for i in range(0, img.shape[0], stride):
        for j in range(0, img.shape[1], stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            yield patch


def unpatch(patched, patch_size):
    rows = []
    for row in patched:
        row_fragments = []
        for pixel in row:
            patch = np.tile(pixel, (patch_size, patch_size, 1))
            row_fragments.append(patch)
        rows.append(np.hstack(row_fragments))
        row_fragments = []
    image = np.vstack(rows)
    if image.shape[2] == 1:
        return np.squeeze(image, axis=2)
    else:
        return image


def patch_image(img, patch_size):
    return patch_map(img, patch_size, lambda p: np.mean(p, axis=(0, 1)))


def patch_groundtruth(img, patch_size):
    return patch_map(img, patch_size, utils.patch_to_class)


def drop_external_layers(img, size_to_drop):
    return img[size_to_drop:-size_to_drop, size_to_drop:-size_to_drop]


def ncwh_to_nwhc(t):
    return t.permute(0,2,3,1)

def nwhc_to_ncwh(t):
    return t.permute(0, 3, 1, 2)


class NCWHtoNWHC(nn.Module):
    def __init__(self):
        super(NCWHtoNWHC, self).__init__()
    def forward(self, x):
        return x.permute(0,2,3,1)


class NWHCtoNCWH(nn.Module):
    def __init__(self):
        super(NWHCtoNCWH, self).__init__()
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class Threshold(nn.Module):
    def __init__(self, val):
        super(Threshold, self).__init__()
        self.val = val
    def forward(self, x):
        s1 = x[:,:,:,0].contiguous().view(-1)
        s2 = x[:,:,:,1].contiguous().view(-1)
        return torch.stack([s1, s2], dim=1)


def mean_shift_filter(img):
    X = np.reshape(img, [-1, 3])
    original_shape = img.shape
    ms = MeanShift(bandwidth=0.013, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    return np.reshape(labels, original_shape[:2])


def reflect_padding(img, border_width):
    padf = lambda img: np.pad(img, mode='reflect', pad_width=border_width)
    if len(img.shape) == 3:
        return np.stack([padf(img[:,:,i]) for i in range(img.shape[2])], axis=-1)
    else:
        return padf(img)


def unstack(img):
    return np.array([img[:,:,c] for c in range(3)])


def restack(img):
    return np.stack(img, axis=-1)

