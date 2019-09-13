from utils.preprocess_utils import split_data, write_hdf5
from utils.format_labels import write_labels_to_csv, dataframe_to_arrray
import pandas as pd

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p","--preprocess")
ap.add_argument("-b","--build_hdf5")
args = vars(ap.parse_args())

BASE_PATH = "../input/"

if args["preprocess"]:
    TRAIN = BASE_PATH + "train.csv"
    TEST = BASE_PATH + "test.csv"
    write_labels_to_csv(TRAIN,TEST)
    PROCESSED_CSV = "../train_sirna_labels.csv"
    data = pd.read_csv(PROCESSED_CSV)
    dataframe_to_arrray(data)
    free_memory(data)


if args["build_hdf5"]:
    HUVEC_VAL = 4000
    HEPG2_VAL= 2000
    RPE_VAL= 2000
    U2OS_VAL= 800
    VAL_DISTRIBUTION = [HUVEC_VAL,HEPG2_VAL,RPE_VAL,U2OS_VAL]
    TEST_DISTRIBUTION = [x/2 for x in VAL_DISTRIBUTION]
    BUILD_DIMS = 512
    BUILD_CHANELS = 6
    CLEAN_PATH = "../clean_data/train"
    TRAIN_HDF5 = "../clean_data/hdf5/train.hdf5"
    VAL_HDF5 = "../clean_data/hdf5/val.hdf5"
    TEST_HDF5 = "../clean_data/hdf5/test.hdf5"
    HDF5_OUTPUTS = [TRAIN_HDF5,VAL_HDF5,TEST_HDF5]

    splits = split_data(TEST_DISTRIBUTION,VAL_DISTRIBUTION,CLEAN_PATH)
    #write_hdf5(splits, BUILD_DIMS,BUILD_CHANELS,HDF5_OUTPUTS)
