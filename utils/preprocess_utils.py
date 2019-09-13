from utils import AspectAwarePreprocessor
from utils import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import progressbar
import json
import cv2
import os
from glob import glob
from pathlib import Path



def pick_subset(values_distribution, list_paths):
    HUVEC = values_distribution[0]
    HEPG2 = values_distribution[1]
    RPE = values_distribution[2]
    U2OS = values_distribution[3]
    print(list_paths[0])


    return "lool"

def split_data(values_distribution,clean_path):
    np.random.seed(107)
    path = Path(clean_path)
    trainPaths = list(path.glob('*.npy'))
    print(len(trainPaths))
    np.random.shuffle(trainPaths)
    subsets = pick_subset(values_distribution,trainPaths)
    # # buff in the format ./dataset/clean_data/data/PNEUMONIA-00.png
    # buff = [x.split("/")[-1] for x in trainPaths]
    # labels = [x.split("-")[0] for x in buff]
    #
    # # le = LabelEncoder()
    # # trainLabels = le.fit_transform(labels)
    #
    # test_split = train_test_split(trainPaths, trainLabels,
	# test_size=num_test_images, stratify=trainLabels,
	# random_state=777)
    #
    # # perform another stratified sampling, this time to build the
    # # validation data
    # val_split = train_test_split(trainPaths, trainLabels,
    # 	test_size=num_val_images, stratify=trainLabels,
    # 	random_state=777)
    # return (test_split, val_split)
    return "Gucci"
def write_hdf5(splits,build_size,output_hdf5s):

    train_hdf5,val_hdf5,test_hdf5 = output_hdf5s
    (_, testPaths, _, testLabels) = splits[0]
    (trainPaths, valPaths, trainLabels, valLabels) = splits[1]

    datasets = [
    	("train", trainPaths, trainLabels, train_hdf5),
    	("val", valPaths, valLabels, val_hdf5),
    	("test", testPaths, testLabels, test_hdf5)]
    aap = AspectAwarePreprocessor(build_size, build_size)

    # loop over the dataset tuples
    for (dType, paths, labels, outputPath) in datasets:
    	# create HDF5 writer
    	print("[INFO] building {}...".format(outputPath))
    	writer = HDF5DatasetWriter((len(paths), build_size, build_size, 3), outputPath)

    	# initialize the progress bar
    	widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
    		progressbar.Bar(), " ", progressbar.ETA()]
    	pbar = progressbar.ProgressBar(maxval=len(paths),
    		widgets=widgets).start()

    	# loop over the image paths
    	for (i, (path, label)) in enumerate(zip(paths, labels)):
    		# load the image and process it
    		image = cv2.imread(path)
    		image = aap.preprocess(image)
    		# add the image and label # to the HDF5 dataset
    		writer.add([image], [label])
    		pbar.update(i)

    	# close the HDF5 writer
    	pbar.finish()
    	writer.close()
