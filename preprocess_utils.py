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

def prepare_dataset(raw_path,clean_path):
    raw_path = Path(raw_path)
    print("There data labels are the following:")
    count = 0
    for case in raw_path.glob('*'):
        label = str(case).split("/")[-1]
        print("Extracting data from ", case)
        for filename in case.glob("*"):
            dst = clean_path + label + "-" + str(count)+".png"
            src =  filename
            os.rename(src,dst)
            count+=1
    print("Extracted a total of",count,"images.")

def check_corrupted_images(clean_path):
    imagePaths = Path(clean_path)
    imagePaths = imagePaths.glob('*.png')
    count = 0
    for image in imagePaths:
        im_vector = cv2.imread(str(image))
        if(im_vector is None):
            print("BAD IMAGE",image)
            os.remove(image)
            continue
        count +=1
    print("Total number of good images is",count)


def split_data(num_test_images, num_val_images,clean_path):
    np.random.seed(107)
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
    	random_state=777)
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
    (R, G, B) = ([], [], [])

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

    		# if we are building the training dataset, then compute the
    		# mean of each channel in the image, then update the
    		# respective lists
    		if dType == "train":
    			(b, g, r) = cv2.mean(image)[:3]
    			R.append(r)
    			G.append(g)
    			B.append(b)

    		# add the image and label # to the HDF5 dataset
    		writer.add([image], [label])
    		pbar.update(i)

    	# close the HDF5 writer
    	pbar.finish()
    	writer.close()

    # construct a dictionary of averages, then serialize the means to a
    # JSON file
    DATASET_MEAN = "./output/malaria_mean.json"

    print("[INFO] serializing means...")
    D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    f = open(DATASET_MEAN, "w")
    f.write(json.dumps(D))
    f.close()
