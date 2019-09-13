import os
import numpy as np
from skimage.io import imread
import pandas as pd

DEFAULT_IMAGES_BASE_PATH = '../input'
DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)


def load_image(image_path):
    im = imread(image_path,as_gray=True).astype(np.uint8)
    return im


def load_images_as_tensor(image_paths, dtype=np.uint8):
    n_channels = len(image_paths)
    data = np.ndarray(shape=(512, 512, n_channels), dtype=dtype)
    for ix, img_path in enumerate(image_paths):
        data[:, :, ix] = load_image(img_path)
    return data



def image_path(dataset, experiment, plate, address, site, channel,
               base_path=DEFAULT_IMAGES_BASE_PATH):
    return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(address, site, channel))

#
def load_site(base_path, dataset, experiment, plate, well, site,
              channels=DEFAULT_CHANNELSs):

    channel_paths = [
        image_path(
            dataset, experiment, plate, well, site, c, base_path=base_path)
        for c in channels
    ]
    return load_images_as_tensor(channel_paths)
