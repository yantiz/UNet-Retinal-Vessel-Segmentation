#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./DATASET/training/images/"
groundTruth_imgs_train = "./DATASET/training/labels/"
borderMasks_imgs_train = "./DATASET/training/masks/"
#test
original_imgs_test = "./DATASET/test/images/"
groundTruth_imgs_test = "./DATASET/test/labels/"
borderMasks_imgs_test = "./DATASET/test/masks/"
#---------------------------------------------------------------------------------------------

shrink_ratio = 0.3
channels = 3
height = 2048
width = 3072
dataset_path = "./DATASET_training_testing/"

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
    Nimgs = len(os.listdir(imgs_dir))
    imgs = np.empty((Nimgs,int(height*shrink_ratio),int(width*shrink_ratio),channels))
    groundTruth = np.empty((Nimgs,int(height*shrink_ratio),int(width*shrink_ratio),channels))
    border_masks = np.empty((Nimgs,int(height*shrink_ratio),int(width*shrink_ratio)))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print ("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i].split('.')[0] + "_labels.tif"
            print ("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            #corresponding border masks
            border_masks_name = files[i].split('.')[0] + "_mask.tif"
            print ("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print ("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,int(height*shrink_ratio),int(width*shrink_ratio)))
    groundTruth = np.transpose(groundTruth,(0,3,1,2))
    assert(groundTruth.shape == (Nimgs,channels,int(height*shrink_ratio),int(width*shrink_ratio)))
    border_masks = np.reshape(border_masks,(Nimgs,1,int(height*shrink_ratio),int(width*shrink_ratio)))
    assert(border_masks.shape == (Nimgs,1,int(height*shrink_ratio),int(width*shrink_ratio)))
    return imgs, groundTruth, border_masks

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
print ("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DATASET_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DATASET_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_path + "DATASET_borderMasks_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
print ("saving test datasets")
write_hdf5(imgs_test,dataset_path + "DATASET_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DATASET_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "DATASET_borderMasks_test.hdf5")
