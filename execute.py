from utils.preprocess_utils import * 
from utils.format_labels import *
import pandas as pd

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p","--preprocess")
ap.add_argument("-f","--free_data")
ap.add_argument("-s","--split")
ap.add_argument("-b","--build")
args = vars(ap.parse_args())

BASE_PATH = "../input/"

if args["preprocess"]:
    TRAIN = BASE_PATH + "train.csv"
    TEST = BASE_PATH + "test.csv"
    write_labels_to_csv(TRAIN,TEST)
    PROCESSED_CSV = "../train_sirna_labels.csv"
    data = pd.read_csv(PROCESSED_CSV)
    dataframe_to_arrray(data)
if["free_data"]:
    PROCESSED_CSV = "../train_sirna_labels.csv"
    data = pd.read_csv(PROCESSED_CSV)
    #free_memory(data)


TRAIN_CSV = "./train_split.csv"
VAL_CSV= "./val_split.csv"
TEST_CSV = "./test_split.csv"
SPLITS_LIST = [TRAIN_CSV, VAL_CSV, TEST_CSV]

if args["split"]:
    HUVEC_VAL = 3600 
    HEPG2_VAL= 1700
    RPE_VAL= 1700
    U2OS_VAL= 750
    VAL_DISTRIBUTION = [HUVEC_VAL,HEPG2_VAL,RPE_VAL,U2OS_VAL]
    TEST_DISTRIBUTION = [x/2 for x in VAL_DISTRIBUTION]
    CLEAN_PATH = "../clean_data/train"
    splits = split_data(TEST_DISTRIBUTION,VAL_DISTRIBUTION,CLEAN_PATH,SPLITS_LIST)
if args["build"]:
    TRAIN_HDF5 = "../clean_data/hdf5/train.hdf5"
    VAL_HDF5 = "../clean_data/hdf5/val.hdf5"
    TEST_HDF5 = "../clean_data/hdf5/test.hdf5"
    HDF5_OUTPUTS = [TRAIN_HDF5,VAL_HDF5,TEST_HDF5]
    train_paths, val_paths, test_paths = load_paths(TRAIN_CSV,VAL_CSV,                                                     TEST_CSV)
    paths = [train_paths,val_paths,test_paths]
    train_labels, val_labels, test_labels= get_labels(paths)
    labels = [train_labels, val_labels, test_labels]
    BUILD_DIMS = 512
    BUILD_CHANELS = 6
    
    #write_hdf5(paths,labels, BUILD_DIMS,BUILD_CHANELS,HDF5_OUTPUTS)
