import numpy as np
from keras.models import load_model
from preprocessing import *
from skimage.color import rgb2hsv
import os, sys

model = load_model('trained_model.hdf5')
TEST_DIR = 'test_set_images/'
test_files = os.listdir(TEST_DIR)
WINDOW_SIZE = 71

for file in test_files:
    filename = '{}{}/{}.png'.format(TEST_DIR, file, file)
    # Load image and convert it to hsv
    img = rgb2hsv(load_image(filename))
    # Add padding so that images pass from (608, 608, 3) to (610, 610, 3)
    img_m = reflect_padding(img, 1)
    # Compress image to obtain (61, 61, 3) and avoid memory errors
    img_m = patch_image(img, 10)
    # Create neighborhoods
    neighborhoods = image_to_neighborhoods(img_m, WINDOW_SIZE, True)
    print('Read and create neighborhoods for', file)
    # Predict
    preds = model.predict(neighborhoods)
    print('Generated predictions')
    # Convert continuous predictions to (0, 1) labels
    preds = (preds[:,1] > preds[:,0]) * 1
    # Reshape to (61, 61)
    preds = np.reshape(preds, img_m.shape[:2])
    # Upsample to (610, 610)
    preds = unpatch(preds, 10)
    # Delete "hack" padding
    preds = preds[1:-1, 1:-1]
    # Make sure prediction has same dimensions as original image
    assert preds.shape == img.shape[:2]
    np.save(file, preds)
    print('Saved array')

