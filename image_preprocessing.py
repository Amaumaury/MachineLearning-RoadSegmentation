import numpy as np
import colorsys
from scipy.ndimage import filters

simple_kernel = np.ones((3, 3))
simple_kernel = simple_kernel / (simple_kernel.shape[0] * simple_kernel.shape[1])

def convolve_image(img, kernel=simple_kernel):
    results = []

    for feature in range(img.shape[2]):
        results.append(np.array(filters.convolve(img[:, :, feature:feature+1].reshape(400, 400), kernel)))

    return np.dstack(results)

def __check_shape(shape):
    X = shape[0]
    Y = shape[1]
    Z = shape[2]

    if X != Y or Z != 3 or len(shape) > 3:
        raise Exception('Unexpected image shape, expecting format (X, X, 3). Got', shape)

    return X, Y, Z


def rgb_to_hsv(img):
    X, Y, Z = __check_shape(img.shape)
    return np.array([colorsys.rgb_to_hsv(pixel[0], pixel[1], pixel[2]) for pixel in img.reshape(X*Y, Z)]).reshape(X, Y, Z)

def saturate_hsv_img(img, saturation_factor=1.5):
    X, Y, Z = __check_shape(img.shape)
    return np.array([[pixel[0], min(1, saturation_factor * pixel[1]), pixel[2]] for pixel in img.reshape(X*Y, Z)]).reshape(X, Y, Z)

def hsv_to_rgb(img):
    X, Y, Z = __check_shape(img.shape)
    return np.array([colorsys.hsv_to_rgb(pixel[0], pixel[1], pixel[2]) for pixel in img.reshape(X*Y, Z)]).reshape(X, Y, Z)

