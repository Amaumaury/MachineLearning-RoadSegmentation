import os, sys
from best_utils import *
import numpy as np

import torch
import torch.nn as nn

from train import train_with_snapshots

ROOT_DIR = 'training/'

IMAGE_DIR = ROOT_DIR + 'images/'

files = os.listdir(IMAGE_DIR)
imgs = [load_image(IMAGE_DIR + file) for file in files]

GT_DIR = ROOT_DIR + 'groundtruth/'
gt_imgs = [load_image(GT_DIR + file) for file in files]

KERNEL_SIDE = 3

features = np.vstack([image_to_features(img, KERNEL_SIDE) for img in imgs[:2]])
labels = [crop_groundtruth(gt, KERNEL_SIDE) for gt in gt_imgs[:2]]
labels = np.ravel(labels)

X = torch.from_numpy(features)
Y = torch.from_numpy(labels)

model = nn.Sequential(
    nn.Linear(KERNEL_SIDE**2 * 3, 12),
    nn.Sigmoid(),
    nn.Linear(12, 1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

costf = torch.nn.MSELoss()

NITER = 10*6

train_with_snapshots(X, Y, model, costf, optimizer, 10**6, 1000, 5000)

