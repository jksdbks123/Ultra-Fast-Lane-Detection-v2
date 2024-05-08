
import torch, os, cv2
from utils.dist_utils import dist_print
import torch, os
from utils.common import merge_config, get_model
from tqdm import tqdm
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from utils.config import Config
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords

def extract_lane_patch(coords,vis,target_img_folder,route_img_name,img_h = 640,img_w = 640):
    for i,lane_coords in enumerate(coords):
        lane_coords = np.array(lane_coords)
        lane_params = np.polyfit(lane_coords[:,0],lane_coords[:,1],1)
        x = np.linspace(0,img_w,100)
        y = lane_params[0] * x + lane_params[1]
        valid_ind = np.where((y >= 2*img_h/3) & (y <= img_h))[0]
        if valid_ind.shape[0] == 0:
            continue
        x = x[valid_ind]
        y = y[valid_ind]
        # create a mask with 20 pixel width for the lane
        mask = np.zeros((img_h,img_w))
        # draw the lane line using cv2.line
        cv2.line(mask,(int(x[0]),int(y[0])),(int(x[-1]),int(y[-1])),1,60)
        lane_patch = cv2.bitwise_and(vis,vis,mask = mask.astype(np.uint8))
        xs,ys = np.where(mask == 1)
        xmin,xmax = xs.min(),xs.max()
        if xmin == xmax:
            continue
        ymin,ymax = ys.min(),ys.max()
        if ymin == ymax:
            continue
        lane_patch = lane_patch[xmin:xmax,ymin:ymax]
        # save the lane patch
        cv2.imwrite(os.path.join(target_img_folder,f'{route_img_name}_{i}.jpg'),lane_patch)