import numpy as np
import os, sys
import csv

import torch
from torch.autograd import Variable
import torch.nn as nn

from best_utils import *
from train import train
from cross_validation import cross_validation
from image_processing import rgb_to_hsv, saturate_hsv_img

ROOT_DIR = 'training/'
IMAGE_DIR = ROOT_DIR + 'images/'

files = os.listdir(IMAGE_DIR)
imgs = np.stack([saturate_hsv_img(rgb_to_hsv(load_image(IMAGE_DIR + file))) for file in files])

GT_DIR = ROOT_DIR + 'groundtruth/'
gt_imgs = np.stack([load_image(GT_DIR + file) for file in files])

CONV_KERNEL_SIDE = 5
PATCH_SIDE = 10

patched_imgs = np.stack([patch_image(im, PATCH_SIDE) for im in imgs])
patched_gts = np.stack([patch_groundtruth(gt, PATCH_SIDE) for gt in gt_imgs])

features_per_image = [image_to_features(img, CONV_KERNEL_SIDE) for img in patched_imgs]
flattened_labels_per_image = [np.ravel(crop_groundtruth(img, CONV_KERNEL_SIDE)) for img in patched_gts]

W, H, *_ = imgs[0].shape
PATCHES_PER_IMAGE = (W * H) // PATCH_SIDE**2

def train_predict(test_indices, train_indices, learning_rate, hidden_size, niter):
    train_x = np.vstack([features_per_image[i] for i in train_indices])
    train_y = np.ravel([flattened_labels_per_image[i] for i in train_indices])

    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y).type(torch.FloatTensor)

    test_x = np.vstack([features_per_image[i] for i in test_indices])
    test_y = np.ravel([np.ravel(crop_groundtruth(gt_imgs[i])) for i in test_indices])
    #test_y = np.ravel([flattened_labels_per_image[i] for i in test_indices])

    test_x = Variable(torch.from_numpy(test_x))

    mlp = nn.Sequential(
        nn.Linear(CONV_KERNEL_SIDE**2 * 3, hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, 1)
    )

    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    costf = torch.nn.MSELoss()
    train(train_x, train_y, mlp, costf, optimizer, niter, 500)

    y_pred = mlp(test_x).data.numpy()

    pred_wide = []
    for i in range(0, len(y_pred), PATCHES_PER_IMAGE):
        curr_pred = y_pred[i: i + PATCHES_PER_IMAGE]
        pred_image = np.reshape(curr_pred, (W // PATCH_SIDE, H // PATCH_SIDE))
        pred_image = np.ravel(unpatch(pred_image, PATCH_SIDE))
        pred_wide.append(pred_image)
    y_pred = np.ravel(pred_wide)

    y_pred[y_pred <= 0] = -1
    y_pred[y_pred > 0] = 1

    return y_pred, test_y


def generate_train_predict_MLP(learning_rate, hidden_size, niter):
    return lambda test_indices, train_indices: train_predict(test_indices, train_indices,
                                                             learning_rate, hidden_size, niter)


learning_rates_candidates = np.logspace(-1, -4, 8, endpoint=True)
hidden_size_candidates = range(9, 44, 3)

f = open('mlp_gridsearch.csv', 'w')
notepad = csv.writer(f)
for hidden_size in hidden_size_candidates:
    for lr in learning_rates_candidates:
        f = generate_train_predict_MLP(lr, hidden_size, 20000)
        res = cross_validation(len(imgs), 5, f)
        print([hidden_size, lr, res])
        notepad.writerow([hidden_size, lr, res])
f.close()

