import os, sys
from best_utils import *
import numpy as np

import torch
import torch.nn as nn

from train import train_with_snapshots

ROOT_DIR = 'training/'
IMAGE_DIR = ROOT_DIR + 'images/'
GT_DIR = ROOT_DIR + 'groundtruth/'
KERNEL_SIDE = 3
HIDDEN_SIZE = 12
LEARNING_RATE = 0.1
NITER = 10**6

model = nn.Sequential(
    nn.Linear(KERNEL_SIDE**2 * 3, HIDDEN_SIZE),
    nn.Sigmoid(),
    nn.Linear(HIDDEN_SIZE, 1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

costf = torch.nn.MSELoss()

if __name__ == '__main__':
    files = os.listdir(IMAGE_DIR)

    # Read images into numpy arrays
    imgs = [load_image(IMAGE_DIR + file) for file in files]

    # Read groundtruths into numpy arrays
    gt_imgs = [load_image(GT_DIR + file) for file in files]

    # Create matrix of features
    features = np.vstack([image_to_features(img, KERNEL_SIDE) for img in imgs])

    # Prepare labels
    labels = [crop_groundtruth(gt, KERNEL_SIDE) for gt in gt_imgs]
    labels = np.ravel(labels)

    # Create pytorch tensors
    X = torch.from_numpy(features)
    Y = torch.from_numpy(labels)

    # train
    train_with_snapshots(X, Y, model, costf, optimizer, NITER, 1000, 5000)

