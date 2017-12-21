import numpy as np
from preprocessing import *
import sys, os
import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LeakyReLU, Dropout

# Image locations
ROOT_DIR = 'training/'
IMAGE_DIR = ROOT_DIR + 'images/'
GT_DIR = ROOT_DIR + 'groundtruth/'

# Preprocessing parameters
PATCH_SIZE = 10
NEIGHBORHOOD_SIZE = 71

# Training parameters
N = 100 # Number of image to be used in training
EPOCHS = 100
LEAKYNESS = 0.1
BATCH_SIZE = 1600
LEARNING_RATE = 0.001

files = os.listdir(IMAGE_DIR)

# Load images and groudtruths
imgs = np.stack([load_image(IMAGE_DIR + file) for file in files]) # images (400, 400, 3)
gt_imgs = np.stack([load_image(GT_DIR + file) for file in files]) # images (400, 400)

# Apply patching compression to images and grondtruths
patched_imgs = np.stack([patch_image(img, PATCH_SIZE) for img in imgs])
patched_gts = np.stack([patch_groundtruth(gt, PATCH_SIZE) for gt in gt_imgs])


# Create neighborhoods
neighborhoods_per_image = [image_to_neighborhoods(im, NEIGHBORHOOD_SIZE, True) for im in patched_imgs[:N]]
neighborhoods = np.vstack(neighborhoods_per_image)

# Prepare 1 label per neighborhood
labels = np.ravel(patched_gts[:N])
labels = keras.utils.np_utils.to_categorical(labels)

# Define model
cnn = keras.models.Sequential([
    Conv2D(32, (5, 5), strides=(1, 1), input_shape=neighborhoods.shape[1:]),
    LeakyReLU(LEAKYNESS),

    MaxPooling2D(2),
    Dropout(0.25),

    Conv2D(64, (3, 3), strides=(1, 1)),
    LeakyReLU(LEAKYNESS),

    MaxPooling2D(2),
    Dropout(0.25),

    Conv2D(128, (3, 3), strides=(1, 1)),
    LeakyReLU(LEAKYNESS),

    MaxPooling2D(2),
    Dropout(0.25),

    Conv2D(256, (3, 3), strides=(1, 1)),
    LeakyReLU(LEAKYNESS),

    MaxPooling2D(2),
    Dropout(0.25),

    Dense(128),
    LeakyReLU(LEAKYNESS),

    Flatten(),
    Dense(2, activation='sigmoid'),
])

# Compile model
cnn.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
        metrics=['accuracy'])

# Log progress to file and periodically save model
callbacks = [
    keras.callbacks.CSVLogger('train_log.csv', separator=',', append=False),
    keras.callbacks.ModelCheckpoint(filepath='train_checkpoint.hdf5', verbose=1, period=20)
]

# Train
cnn.fit(neighborhoods, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

# Save trained model
cnn.save('trained_model.hdf5')

