from preprocess_utils import prepare_dataset, check_corrupted_images, split_data, write_hdf5
from utils.format_labels import write_labels_to_csv, dataframe_to_arrray
import pandas as pd


BASE_PATH = "../input/"
TRAIN = BASE_PATH + "train.csv"
TEST = BASE_PATH + "test.csv"
# write_labels_to_csv(TRAIN,TEST)


LABELED = "../train_sirna_labels.csv"
data = pd.read_csv(LABELED)
dataframe_to_arrray(data)
################
NUM_TEST_IMAGES = 2700
NUM_VAL_IMAGES = 2700
CLEAN_PATH = "../clean_data/data/"


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
