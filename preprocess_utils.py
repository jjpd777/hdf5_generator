from utils import config
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
from Pathlib import Path

def prepare_dataset(raw_path,clean_path):
    data_folders = ["train/","test/","val/"]

    sub_folder =list(glob(raw_path + "train/*"))
    labels = [x.split("/")[-1] for x in sub_folder]
    print("There data labels are the following:")
    print(labels[0],labels[1])

    count = 0
    for directory in data_folders:
        directory_in_use = MAIN_PATH + directory
        for case in labels:
            case_dir = directory_in_use + case + "/"
            print("Extracting data from ", case_dir)
            for filename in os.listdir(case_dir):
                dst = clean_path + case+"-"+ str(count)+".jpeg"
                src = case_dir +filename
                os.rename(src,dst)
                count+=1
    print("Extracted a total of",count,"images.")

def check_corrupted_images(clean_path):
    imagePaths = Path(clean_path)
    imagePaths = imagePaths.glob('*.png')
    count = 0
    for image in imagePaths:
        im_vector = cv2.imread(image)
        if(im_vector is None):
            print("BAD IMAGE",image)
            os.remove(image)
            continue
        count +=1
    print("Total number of good images is",count)


def split_data(num_test_images, num_val_images,clean_path):
    trainPaths = list(paths.list_images(clean_path))
    np.shuffle(trainPaths)
    # buff in the format ./dataset/clean_data/data/PNEUMONIA-00.png
    buff = [x.split("/")[-1] for x in trainPaths]
    labels = [x.split("-")[0] for x in buff]
    le = LabelEncoder()
    trainLabels = le.fit_transform(labels)

    test_split = train_test_split(trainPaths, trainLabels,
	test_size=num_test_images, stratify=trainLabels,
	random_state=777)

    # perform another stratified sampling, this time to build the
    # validation data
    val_split = train_test_split(trainPaths, trainLabels,
    	test_size=num_val_images, stratify=trainLabels,
    	random_state=777,shuffle=True)
    return (test_split, val_split)

def write_hdf5(splits,build_size,output_hdf5s):

    train_hdf5,val_hdf5,test_hdf5 = output_hdf5s
    (trainPaths, testPaths, trainLabels, testLabels) = splits[0]
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

