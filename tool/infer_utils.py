import os
import numpy as np
import cv2
from skimage import morphology
import torch

def cam_npy_to_cam_dict(cam_np, label):
    cam_dict = {}
    idxs = np.where(label==1)[0]
    for idx in idxs:
        cam_dict[idx] = cam_np[idx]
    return cam_dict

def dict2npy(cam_dict, gt_label, orig_img):

    gt_cat = np.where(gt_label==1)[0]
    if len(gt_cat) == 0:
        return np.zeros((2,) + orig_img.shape[:2])
    orig_img_size = cam_dict[gt_cat[0]].shape
    cam_npy = np.zeros((2, orig_img_size[0], orig_img_size[1]))
    # for gt in gt_cat:
        # if gt == 0:  
        #     cam_npy[1] = cam_dict[gt]  
        # else:  
        #     cam_npy[0] = cam_dict[gt]  
    for gt in gt_cat:
        cam_npy[gt] = cam_dict[gt]
    return cam_npy

def gen_bg_mask(orig_img):
    img_array = np.array(orig_img).astype(np.uint8)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    binary = np.uint8(binary)    
    dst = morphology.remove_small_objects(binary==255,min_size=50,connectivity=1)
    bg_mask = np.zeros(orig_img.shape[:2])
    bg_mask[dst==True]=1.000001
    return bg_mask

def cam_npy_to_label_map(cam_npy):
    seg_map = cam_npy.transpose(1,2,0)
    # 使用argmax后反转结果，使CAM较大的区域对应1，较小的区域对应0
    seg_map = np.asarray(np.argmax(seg_map, axis=2), dtype=int)
    return seg_map

def cam_npy_to_cam_dict(cam_npy, label):
    cam_dict = {}
    for i in range(len(label)):
        if label[i] > 1e-5:
            cam_dict[i] = cam_npy[i]
    return cam_dict





















