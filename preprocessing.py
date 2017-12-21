import matplotlib.image as mpimg

import numpy as np

from sklearn.cluster import MeanShift


def load_image(filename):
    """Loads an image into a numpy array"""
    return mpimg.imread(filename)


def neighborhood_iter(img, kernel_size, pad):
    """Iterator over the neighborhoods of the provided image. If pad is True,
    reflect padding is added
    """
    v = (kernel_size - 1) // 2
    if pad:
        img = reflect_padding(img, v)
    for i in range(img.shape[0] - (kernel_size - 1)):
        for j in range(img.shape[1] - (kernel_size - 1)):
            yield img[i : i + kernel_size, j : j + kernel_size]


def image_to_neighborhoods(img, kernel_size, pad):
    """Converts image in tensor of neighborhoods"""
    return np.stack(list(neighborhood_iter(img, kernel_size, pad)))


def image_to_features(img, kernel_size, pad):
    """Linearizes patches of an image into lines.
    Arguments:
     :img: the image to linearize of shape (W, H, C)
     :kernel_size: the length of the side of the patch which will be squared
     should be odd.

    The radius of the patch is r = (kernel_size - 1) / 2
    The produced matrix has shape ((W * H, kernel_size**2 * C)
    """
    return np.vstack(list(map(np.ravel, neighborhood_iter(img, kernel_size, pad))))


def reassemble(lines, kernel_size, original_w, original_h, channels=3):
    """Reassembles the matrix produced by image_to_features into an image"""
    v = (kernel_size - 1) // 2
    center_flat = v * kernel_size + v
    shifter = np.arange(lines.shape[0]) * lines.shape[1]
    shifter = np.reshape(shifter, (len(shifter), 1))
    shifter = np.tile(shifter, (1, channels))
    mask = np.array([center_flat * channels + i for i in range(channels)])
    mask = np.tile(mask, (lines.shape[0], 1))
    mask = shifter + mask
    mask = np.ravel(mask)
    flat = np.ravel(lines)
    pixels = flat[mask]
    return np.reshape(pixels, (original_w, original_h, channels))


def crop_groundtruth(img):
    """Converts the continuous values in the groundtruths images into discrete
    0 and 1"""
    img[img < 0.5] = -1
    img[img >= 0.5] = 1
    return img


def patch_map(img, patch_size, f=lambda p: p):
    """Downsamples the provided image by moving a squared patch of size
    patch_size over it and applying, for each position, the function f.
    Each pixel is used only once (strid = patch_size)
    """
    rows = []
    for i in range(0, img.shape[0], patch_size):
        row = []
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            row.append(f(patch))
        rows.append(row)
    return np.array(rows)


def unpatch(patched, patch_size):
    """Upsamples the provided image. Each pixel is repeated in square of dimension
    (patch_size, patch_size).
    """
    rows = []
    for row in patched:
        row_fragments = []
        for pixel in row:
            patch = np.tile(pixel, (patch_size, patch_size, 1))
            row_fragments.append(patch)
        rows.append(np.hstack(row_fragments))
        row_fragments = []
    image = np.vstack(rows)
    if image.shape[2] == 1:
        return np.squeeze(image, axis=2)
    else:
        return image


def patch_image(img, patch_size):
    """Compresses image using patching with patch size patch_size"""
    return patch_map(img, patch_size, lambda p: np.mean(p, axis=(0, 1)))


def patch_groundtruth(img, patch_size):
    """Compresses groundtruth using patchin with patch_size patch_size"""
    return patch_map(img, patch_size, patch_to_class)


def reflect_padding(img, border_width):
    """Pads image or groundtruth using mirroring"""
    padf = lambda img: np.pad(img, mode='reflect', pad_width=border_width)
    if len(img.shape) == 3:
        return np.stack([padf(img[:,:,i]) for i in range(img.shape[2])], axis=-1)
    else:
        return padf(img)


def patch_to_class(patch, threshold=0.25):
    """Indicates whether the provided patch of groundtruth is road or not by
    comparing the percentage of road pixels in the patch to the provided
    threshold"""
    return 1 if np.mean(patch) > threshold else 0

