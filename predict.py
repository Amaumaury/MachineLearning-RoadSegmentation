import numpy as np
from keras.models import load_model
from best_utils import *
import os, sys

m = load_model('trained_model.hdf5')
TEST_DIR = 'test_set_images/'
test_files = os.listdir(TEST_DIR)
WINDOW_SIZE = 71


for file in test_files:
    filename = '{}{}/{}.png'.format(TEST_DIR, file, file)
    img = load_image(filename)
    img_m = reflect_padding(img, 1)
    img_m = patch_image(img, 10)
    windows = image_to_features(img_m, WINDOW_SIZE, True)
    print('Read and windowed', file)
    preds = m.predict(windows)
    print('Generated predictions')
    preds = (preds[:,1] > preds[:,0]) * 1
    preds = np.reshape(preds, img_m.shape[:2])
    preds = unpatch(preds, 10)
    preds = preds[1:-1, 1:-1]
    assert preds.shape == img.shape[:2]
    np.save(file, preds)
    print('Saved array')
