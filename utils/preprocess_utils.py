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


def select_sister_sirna(original_data,val_test_subset):
    ## Receives original_data, val_test_subset
    ## Eliminate repeated items
    ## gets sister SiRNA
    ## return val_test_subset, remaining data
    sister_sirna = []
    [original_data.remove(x) for x in val_test_subset]
    for image in val_test_subset:
       substring = image[:-6]
       for og in original_data:
           if substring in og:
               sister_sirna.append(og)
    result_val_test = val_test_subset + sister_sirna 
    return (original_data, result_val_test)

    return
def pick_subset(values_distribution, image_paths):
    HUVEC = values_distribution[0]
    HEPG2 = values_distribution[1]
    RPE = values_distribution[2]
    U2OS = values_distribution[3]
    image_paths = [str(x) for x in image_paths]
    subsets = [("HUVEC", HUVEC), ("HEPG2", HEPG2),
               ("RPE", RPE), ("U2OS", U2OS)]
    val_test_subset= []
    for(label, num) in subsets:
        ptr = 0
        label_count = 0
        while(num//2>label_count):
            if(label in image_paths[ptr]):
                val_test_subset.append(image_paths[ptr])
                label_count+=1
                ptr+=1
            else:
                ptr+=1
        print("This is label",label,"with a total count of",label_count)
    print("This is the total subset prior to sister matching", len(val_test_subset))
    original_data, subset = select_sister_sirna(image_paths,val_test_subset)
    print("This is the total subset after sister matching", len(subset))


    return (original_data,subset) 

def split_data(test_distribution,val_distributions,clean_path):
    np.random.seed(107)
    path = Path(clean_path)
    trainPaths = list(path.glob('*.npy'))
    np.random.shuffle(trainPaths)
    print("[INFO] Processing test data..")
    train_data, test_data= pick_subset(test_distribution,trainPaths)
    print("[INFO] Processing validation data..")
    train_data, val_data= pick_subset(val_distributions,train_data)
    result = train_data + val_data + test_data
    print(len(result))
    return (test_split, val_split)
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
