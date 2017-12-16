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
    windows = image_to_features(img, WINDOW_SIZE, True)
    print('Read and windowed', file)
    preds = m.predict(windows)
    print('Generated predictions')
    preds = (preds[:,1] > preds[:,0]) * 1
    preds = np.reshape(preds, img.shape[:2])
    np.save(file, preds)
    print('Saved array')
