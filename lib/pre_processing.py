###################################################
#
#   Script to pre-process the original imgs
#
##################################################


import numpy as np
from PIL import Image
import cv2

from help_functions import *
from utils.image_preprocessing import correct_illumination

#My pre processing for RGB images (use for both training and testing!)
def my_PreProc_RGB(data, param=None):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #my preprocessing:
    #train_imgs = ben_preprocessing(data)
    #train_imgs = histo_equalized_RGB(data)
    #train_imgs = illumination_correction(data)
    train_imgs = dataset_standardized(data, param)
    return train_imgs


#My pre processing for grayscale images (use for both training and testing!)
def my_PreProc_gray(data, param=None):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #my preprocessing:
    #train_imgs = illumination_correction(data)
    train_imgs = rgb2gray(data)
    #train_imgs = histo_equalized(train_imgs)
    train_imgs = dataset_standardized(train_imgs, param)
    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#==== histogram equalization
def ben_preprocessing(imgs, sigmaX=10):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==3)  #check the channel is 3
    imgs_preprocessed = np.transpose(imgs,(0,2,3,1))
    for i in range(imgs_preprocessed.shape[0]):
        img = np.uint8(cv2.normalize(imgs_preprocessed[i], None, 0, 255, cv2.NORM_MINMAX))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
        imgs_preprocessed[i] = img.astype(imgs_preprocessed.dtype)
    imgs_preprocessed = np.transpose(imgs_preprocessed,(0,3,1,2))
    return imgs_preprocessed

def illumination_correction(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==3)  #check the channel is 3
    imgs_corrected = np.transpose(imgs,(0,2,3,1))
    for i in range(imgs_corrected.shape[0]):
        img = np.uint8(cv2.normalize(imgs_corrected[i], None, 0, 255, cv2.NORM_MINMAX))
        img = correct_illumination(img)
        imgs_corrected[i] = img.astype(imgs_corrected.dtype)
    imgs_corrected = np.transpose(imgs_corrected,(0,3,1,2))
    return imgs_corrected

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


#==== histogram equalization for RGB images
def histo_equalized_RGB(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==3)  #check the channel is 3
    imgs_equalized = np.transpose(imgs,(0,2,3,1))
    for i in range(imgs_equalized.shape[0]):
        img = np.uint8(cv2.normalize(imgs_equalized[i], None, 0, 255, cv2.NORM_MINMAX))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
        imgs_equalized[i] = img.astype(imgs_equalized.dtype)
    imgs_equalized = np.transpose(imgs_equalized,(0,3,1,2))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== standardize over the dataset for each channel
def dataset_standardized(imgs, param):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1 or imgs.shape[1]==3)  #check the channel is 1 or 3
    imgs_standardized = imgs
    param_save = dict()
    for c in range(imgs_standardized.shape[1]):
        imgs_std = imgs_mean = None
        if param is None: 
            imgs_std = np.std(imgs_standardized[:,c])
            imgs_mean = np.mean(imgs_standardized[:,c])
            param_save[c] = [imgs_std, imgs_mean]
        else:
            imgs_std = param[c][0]
            imgs_mean = param[c][1]

        imgs_standardized[:,c] = (imgs_standardized[:,c]-imgs_mean)/imgs_std

    if param is None:
        np.save("test/param.npy", param_save)

    return imgs_standardized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs
