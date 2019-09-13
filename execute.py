from preprocess_utils import prepare_dataset, check_corrupted_images, split_data, write_hdf5
from utils.format_labels import write_labels_to_csv, dataframe_to_arrray
import pandas as pd

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p","--preprocess")
ap.add_argument("-b","--build_hdf5")
args = vars(ap.parse_args())

BASE_PATH = "../input/"
TRAIN = BASE_PATH + "train.csv"
TEST = BASE_PATH + "test.csv"

if args["preprocess"]:
    write_labels_to_csv(TRAIN,TEST)
    PROCESSED_CSV = "../train_sirna_labels.csv"
    data = pd.read_csv(PROCESSED_CSV)
    dataframe_to_arrray(data)
    free_memory(data)

NUM_TEST_IMAGES = 2700
NUM_VAL_IMAGES = 2700
CLEAN_PATH = "../clean_data/train/"
if args["build_hdf5"]:
################


#prepare_dataset(RAW_PATH, CLEAN_PATH)
#check_corrupted_images(CLEAN_PATH)
splits = split_data(NUM_TEST_IMAGES, NUM_VAL_IMAGES,
                    CLEAN_PATH)

BUILD_SIZE = 256
TRAIN_HDF5 = "../clean_data/hdf5/train.hdf5"
VAL_HDF5 = "../clean_data/hdf5/val.hdf5"
TEST_HDF5 = "../clean_data/hdf5/test.hdf5"
HDF5_OUTPUTS = [TRAIN_HDF5,VAL_HDF5,TEST_HDF5]
write_hdf5(splits, BUILD_SIZE ,HDF5_OUTPUTS)
