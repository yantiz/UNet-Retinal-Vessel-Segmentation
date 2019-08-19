import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.measure import label

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img


#prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==3 )  #check the channel is 3
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],masks.shape[1],im_h*im_w))
    new_masks = np.transpose(masks,(0,2,1))
    return new_masks


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,3)
    assert (pred.shape[2]==3 )  #check the classes are 3
    if mode=="original":
        pass
    elif mode=="threshold":
        pred[pred >= (1/3.0)] = 1
        pred[pred < (1/3.0)] = 0
    else:
        print "mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'"
        exit()
    pred = np.transpose(pred,(0,2,1))
    pred_images = np.reshape(pred,(pred.shape[0], 3, patch_height, patch_width))
    return pred_images


def hard_pred(pred_img):
    pred_hard = np.copy(pred_img)

    for x in range(pred_hard.shape[1]):
        for y in range(pred_hard.shape[2]):
            channels = pred_hard[:,x,y]
            maximums = np.argwhere(channels == np.amax(channels)) 
            if maximums.shape[0] > 1:
                # There exists tie:
                channels[:] = 0
                channels[1] = 1
            else:
                channels[:] = 0
                channels[maximums[0,0]] = 1
    return pred_hard


def smooth_pred(pred_img):
    pred_smoothed = np.copy(pred_img)
    pred_smoothed = np.transpose(pred_smoothed, (1, 2, 0))

    mask_artery, mask_background, mask_vein = pred_smoothed[:,:,0] == 1, pred_smoothed[:,:,1] == 1, pred_smoothed[:,:,2] == 1

    pred_flat = np.zeros(pred_smoothed.shape[:2])
    pred_flat[mask_background] = 1
    pred_flat[mask_vein] = 2

    pred_binary = np.zeros(pred_smoothed.shape[:2])
    pred_binary[mask_artery] = 1
    pred_binary[mask_vein] = 1

    components = label(pred_binary, neighbors=4)

    for l in np.unique(components):
        mask_l = (components == l)
        uniques, counts = np.unique(pred_flat[mask_l], return_counts=True)
        l_num = uniques[np.argmax(counts)]
        if l_num == 0:
            pred_smoothed[mask_l,:] = np.array([1, 0, 0])
        elif l_num == 2:
            pred_smoothed[mask_l,:] = np.array([0, 0, 1])
        else:
            pred_smoothed[mask_l,:] = np.array([0, 1, 0])
    
    return np.transpose(pred_smoothed, (2, 0, 1))