from utils.preprocess_utils import prepare_dataset, check_corrupted_images, split_data, write_hdf5

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p","--preprocess",type=int,default=0)
ap.add_argument("-b","--build",type=int,default=0)
args = vars(ap.parse_args())

NUM_TEST_IMAGES = 2700
NUM_VAL_IMAGES = 2700
BUILD_SIZE = 256
TRAIN_HDF5 = "../clean_data/hdf5/train.hdf5"
VAL_HDF5 = "../clean_data/hdf5/val.hdf5"
TEST_HDF5 = "../clean_data/hdf5/test.hdf5"
HDF5_OUTPUTS = [TRAIN_HDF5,VAL_HDF5,TEST_HDF5]

if args["preprocess"]:
    RAW_PATH = "../cell_images/"
    CLEAN_PATH = "../clean_data/data/"
    prepare_dataset(RAW_PATH, CLEAN_PATH)
    check_corrupted_images(CLEAN_PATH)
if args["build"]:
    splits = split_data(NUM_TEST_IMAGES, NUM_VAL_IMAGES,
                        CLEAN_PATH)
    write_hdf5(splits, BUILD_SIZE ,HDF5_OUTPUTS)
