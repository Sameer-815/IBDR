import numpy as np
import torch
import itertools
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms

def infer(model, dataroot, n_class, args):
    model.eval()
    model = model.cuda()
    cam_list = []
    gt_list = []
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot, 'img/'), transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=False)
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]
        img_path = os.path.join(dataroot + '/img/' + img_name + '.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]
        def _work(i, img, thr=0.25): 
            with torch.no_grad():
                img = img.cuda()
                cam1, cam2, cam3, y = model.forward_cam(img)
                y = y.cpu().detach().numpy().tolist()[0]
                label = torch.tensor([1.0 if j > thr else 0.0 for j in y])
                cam3 = F.interpolate(cam3, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam1 = F.interpolate(cam1, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam2 = F.interpolate(cam2, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam = cam = cam1 + cam2 + cam3
                cam = cam.cpu().numpy() * label.clone().view(2, 1, 1).numpy()
                return cam, label
        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam - np.min(sum_cam)) / (np.max(sum_cam) - np.min(sum_cam))
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score = infer_utils.dict2npy(cam_dict, label, orig_img)
        seg_map = infer_utils.cam_npy_to_label_map(cam_score)
        if iter % 100 == 0:
            print(iter)   
        gt_map_path = os.path.join(os.path.join(dataroot, 'mask/'), img_name + '.png')
        gt_map = np.array(Image.open(gt_map_path))
        gt_map = 1-(gt_map == 1).astype(np.uint8)  
        if seg_map.ndim == 3:
            seg_map = seg_map[..., 0]
        if gt_map.ndim == 3:
            gt_map = gt_map[..., 0]  
        cam_list.append(seg_map)  
        gt_list.append(gt_map)
        
    return iouutils.scores(gt_list, cam_list, n_class=n_class)

def get_mask(model, dataroot, args, save_path):
    if args.dataset == 'digestpath':
        palette = [239, 239, 239, 63, 125, 88]  
        palette += [0] * (768 - len(palette))
    elif args.dataset == 'wien':
        palette = [0, 0, 0, 255, 255, 255]  
        palette += [0] * (768 - len(palette))
    elif args.dataset == 'glas':
        palette = [0, 0, 0, 255, 255, 255]  
        palette += [0] * (768 - len(palette))
    elif args.dataset == 'hph':
        palette = [0, 0, 0, 255, 255, 255]  
        palette += [0] * (768 - len(palette))
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(data_path=dataroot, transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=False)
    model = model.cuda()
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]
        img_basename = os.path.basename(img_name) 
        img_path = os.path.join(dataroot, img_basename + '.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]
        def _work(i, img, thr=0.25):
            with torch.no_grad():
                img = img.cuda()  
                cam1, cam2, cam3, y = model.forward_cam(img)
                y = y.cpu().detach().numpy().tolist()[0]
                label = torch.tensor([1.0 if j > thr else 0.0 for j in y])
                cam3 = F.interpolate(cam3, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam1 = F.interpolate(cam1, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam2 = F.interpolate(cam2, orig_img_size, mode='bilinear', align_corners=False)[0]
                if args.dataset == 'digestpath':
                    cam = cam1 + cam2 + cam3
                if args.dataset == 'wien':
                    cam = cam1 + cam2 + cam3
                if args.dataset == 'glas':
                    cam = cam1 + cam2 + cam3
                if args.dataset == 'hph':
                    cam = cam1 + cam2 + cam3
                cam = cam.cpu().numpy() * label.clone().view(2, 1, 1).numpy()
                return cam, label
        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam - np.min(sum_cam)) / (np.max(sum_cam) - np.min(sum_cam))
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score = infer_utils.dict2npy(cam_dict, label, orig_img)
        seg_map = infer_utils.cam_npy_to_label_map(cam_score)
        seg_map = 1 - seg_map
        visualimg = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(save_path, img_basename + '.png'), format='PNG')  
        if iter % 100 == 0:
            print(iter)


