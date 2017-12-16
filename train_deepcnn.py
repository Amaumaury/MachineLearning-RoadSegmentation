import numpy as np
from best_utils import *
import sys, os
import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LeakyReLU, Dropout

ROOT_DIR = 'training/'
IMAGE_DIR = ROOT_DIR + 'images/'
GT_DIR = ROOT_DIR + 'groundtruth/'

PATCH_SIZE = 10

WINDOW_SIZE = 71

files = os.listdir(IMAGE_DIR)

imgs = np.stack([load_image(IMAGE_DIR + file) for file in files]) # images (400, 400, 3)
gt_imgs = np.stack([load_image(GT_DIR + file) for file in files]) # images (400, 400)

patched_imgs = np.stack([patch_image(img, PATCH_SIZE) for img in imgs]) # images (400, 400)
patched_gts = np.stack([patch_groundtruth(gt, PATCH_SIZE) for gt in gt_imgs])

PATCHED_SIZE = imgs.shape[1] // PATCH_SIZE
WINDOWS_PER_IMAGE = PATCHED_SIZE ** 2

N = 1 # Number of image to be used in training
epochs = 10

leakyness = 0.1

windows_per_image = [image_to_features(im, WINDOW_SIZE, True) for im in patched_imgs[:N]]
windows = np.vstack(windows_per_image)

window_labels = np.ravel(patched_gts[:N])
assert window_labels.shape[0] == windows.shape[0]

window_labels = keras.utils.np_utils.to_categorical(window_labels)

window_cnn = keras.models.Sequential([

    Conv2D(32, (5, 5), strides=(1, 1), input_shape=windows.shape[1:]),
    LeakyReLU(leakyness),

    MaxPooling2D(2),
    Dropout(0.25),

    Conv2D(64, (3, 3), strides=(1, 1)),
    LeakyReLU(leakyness),

    MaxPooling2D(2),
    Dropout(0.25),

    Conv2D(128, (3, 3), strides=(1, 1)),
    LeakyReLU(leakyness),

    MaxPooling2D(2),
    Dropout(0.25),

    Conv2D(256, (3, 3), strides=(1, 1)),
    LeakyReLU(leakyness),

    MaxPooling2D(2),
    Dropout(0.25),

    Dense(128),
    LeakyReLU(leakyness),

    Flatten(),
    Dense(2, activation='sigmoid'),
    #LeakyReLU(leakyness),
])

window_cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

callbacks = [
    keras.callbacks.CSVLogger('train_log.csv', separator=',', append=False),
    keras.callbacks.ModelCheckpoint(filepath='train_checkpoint.hdf5', verbose=1, period=20)
]
window_cnn.fit(windows, window_labels, epochs=epochs, batch_size=1600, callbacks=callbacks)
window_cnn.save('trained_model.hdf5')

