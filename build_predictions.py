import os, argparse
import ConfigParser
import numpy as np
from PIL import Image

from keras.models import model_from_json
from keras.models import Model

import sys
sys.path.insert(0, './lib/')

from pre_processing import my_PreProc_RGB
from extract_patches import paint_border_overlap, extract_ordered_overlap, recompone_overlap
from help_functions import pred_to_imgs, hard_pred, smooth_pred

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--type_ims', metavar='type', default='rgb', choices=['rgb', 'gray'],
                        help='Type of images to be fed.')
    parser.add_argument('--path_ims', metavar='path', required=False, default='retinal_images',
                        help='Images for which we want the A/V predictions to be generated.')
    parser.add_argument('--path_mat', metavar='path', required=False, default='mat_files',
                        help='MATLAB files for images.')
    parser.add_argument('--path_out', metavar='path', required=False, default='results', help='Path for saving results.')
    args = parser.parse_args()

    return args

def get_data(img_path, path_to_model):
    img_names = os.listdir(img_path)
    N_imgs = len(img_names)
    assert (N_imgs > 0)

    imgs = np.empty((N_imgs, 3, 614, 921))

    for i, img_name in enumerate(img_names):
        image = Image.open(os.path.join(img_path, img_name)).convert('RGB')
        img = np.array(image.resize((921, 614), Image.LANCZOS))
        img = np.transpose(img, (2, 0, 1))
        imgs[i] = img
        
    param = None
    if os.path.exists(path_to_model + "/param.npy"):
        param = np.load(path_to_model + "/param.npy", allow_pickle=True).item()
    imgs = my_PreProc_RGB(imgs, param)

    return imgs, img_names


if __name__ == '__main__':
    args = parse_args()

    type_test_ims = args.type_ims
    path_test_ims = args.path_ims
    path_test_mat = args.path_mat
    path_out_ims = args.path_out

    if not os.path.exists(path_test_ims): 
        os.makedirs(path_test_ims)
    if not os.path.exists(path_test_mat): 
        os.makedirs(path_test_mat)
    if not os.path.exists(path_out_ims):
        os.makedirs(path_out_ims)

    path_to_model = 'model_' + type_test_ims

    model = model_from_json(open(path_to_model + '/architecture.json').read())
    model.load_weights(path_to_model + '/weights.h5')

    imgs, img_names = get_data(path_test_ims, path_to_model)

    config = ConfigParser.RawConfigParser()
    config.read(path_to_model + '/configuration.txt')

    patch_h = int(config.get('data attributes', 'patch_height'))
    patch_w = int(config.get('data attributes', 'patch_width'))
    stride_h = int(config.get('testing settings', 'stride_height'))
    stride_w = int(config.get('testing settings', 'stride_width'))

    test_imgs = paint_border_overlap(imgs, patch_h, patch_w, stride_h, stride_w)
    new_h, new_w = test_imgs.shape[2], test_imgs.shape[3]

    for i in range(test_imgs.shape[0]):
        test_img = test_imgs[i]

        patches_img = extract_ordered_overlap(test_img[np.newaxis, ...], patch_h, patch_w,stride_h,stride_w)
        predictions = model.predict(patches_img, batch_size=32, verbose=2)

        pred_patches = pred_to_imgs(predictions, patch_h, patch_w, "original")
        pred_img = recompone_overlap(pred_patches, new_h, new_w, stride_h, stride_w)
        pred_img = pred_img[0,:,0:614,0:921]

        img_name = img_names[i]
        orig_img = np.array(Image.open(os.path.join(path_test_ims, img_name)).convert('RGB'))
        img_name = img_name.split('.')

        pred_img = hard_pred(pred_img)
        #pred_img = smooth_pred(pred_img)
        pred_img = np.transpose(pred_img*255, (1, 2, 0)).astype(np.uint8)

        pred_image = Image.fromarray(pred_img, 'RGB').resize((orig_img.shape[1], orig_img.shape[0]), Image.NEAREST)
        pred_image.save(os.path.join(path_out_ims, img_name[0] + '_pred.' + img_name[1]))

        pred_img = np.array(pred_image)
        pred_background = pred_img[:,:,1]
        mask = pred_background > 0
        pred_img[mask] = 0

        overlay = orig_img + pred_img
        overlay[overlay < orig_img] = 255

        overlay = Image.fromarray(overlay, 'RGB')
        overlay.save(os.path.join(path_out_ims, img_name[0] + '_overlay.' + img_name[1]))