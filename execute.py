from preprocess_utils import prepare_dataset, check_corrupted_images, split_data, write_hdf5 

NUM_TEST_IMAGES = 2700
NUM_VAL_IMAGES = 2700
RAW_PATH = "../cell_images/"
CLEAN_PATH = "../clean_data/data/"

# 
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
