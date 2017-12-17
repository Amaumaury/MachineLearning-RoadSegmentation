import numpy as np
import os, sys
import csv
from preprocessing import *
from cross_validation import cross_validation
import keras
from keras.layers import Dense, LeakyReLU


ROOT_DIR = 'training/'
IMAGE_DIR = ROOT_DIR + 'images/'
GT_DIR = ROOT_DIR + 'groundtruth/'

PATCH_SIZE = 10
N = 100 # Number of image to be used in training

# Load images and groundtruths, patch them
files = os.listdir(IMAGE_DIR)

imgs = np.stack([load_image(IMAGE_DIR + file) for file in files]) # images (400, 400, 3)
gt_imgs = np.stack([load_image(GT_DIR + file) for file in files]) # images (400, 400)

patched_imgs = np.stack([patch_image(img, PATCH_SIZE) for img in imgs]) # images (400, 400)
patched_gts = np.stack([patch_groundtruth(gt, PATCH_SIZE) for gt in gt_imgs])

print('Read and patched images')

PATCHED_SIZE = imgs.shape[1] // PATCH_SIZE

def generate_model(hidden_size, lr, window_size):
    mlp = keras.models.Sequential([
        Dense(hidden_size, input_shape=(window_size**2 * 3,), activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(loss=keras.losses.mean_squared_error,
                optimizer=keras.optimizers.Adam(lr=lr),
                metrics=['accuracy'])
    return mlp

labels = np.ravel(patched_gts[:N])

for ws in range(15, 75, 10):
    matrix_chunks = [image_to_features(im, ws, True) for im in patched_imgs[:N]]
    matrix = np.vstack(matrix_chunks)
    for hidden_size in range(12, 50, 8):
        for rate in [10**(-1), 10**(-2), 10**(-3), 10**(-4)]:

            def train_predict_f(test_i, train_i):
                mod = generate_model(hidden_size, rate, ws)
                mod.fit(matrix[train_i], labels[train_i], epochs=100, batch_size=800)

                preds = mod.predict(matrix[test_i])
                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1
                return preds, labels[test_i]

        f = open('mlp_gridsearch.csv', 'a')
        notepad = csv.writer(f)
        accs = cross_validation(matrix.shape[0], 3, train_predict_f)
        print([hidden_size, rate, ws, accs])
        notepad.writerow([hidden_size, rate, ws, accs])
        f.close()

