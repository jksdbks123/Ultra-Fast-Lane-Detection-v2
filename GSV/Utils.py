import torch, os, cv2
from torchvision.models import resnet18
import torch.nn as nn
from utils.common import get_model
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.config import Config
import numpy as np
from PIL import Image
import geopandas
import pandas as pd
import matplotlib.pyplot as plt
from shapely import Point
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def extract_lane_patch(coords,vis,img_h = 640,img_w = 640):
    patches = []
    valid_coords = []
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
        patches.append(lane_patch)
        valid_coords.append(np.c_[x,y])

    return patches,valid_coords

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

def get_lane_detector(config_file_path,ckpt_path,device):
    cfg = Config.fromfile(config_file_path)
    net_lane = get_model(cfg)
    cfg.test_model = ckpt_path
    cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net_lane.load_state_dict(compatible_state_dict, strict=False)
    net_lane.to(device)
    net_lane.eval()
    img_transforms_lane = transforms.Compose([
            transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return net_lane,img_transforms_lane


from skimage import exposure
def adaptive_equalization(img):
    img = np.array(img)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return Image.fromarray((img_adapteq * 255).astype(np.uint8))

def get_rumble_strip_classifier(ckpt_path,device):

    model_patch = resnet18(pretrained=True)
    num_classes = 2
    model_patch.fc = nn.Linear(model_patch.fc.in_features, num_classes)
    model_patch.load_state_dict(torch.load(ckpt_path))
    model_patch.to(device)
    model_patch.eval()

    transform_patch = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.Lambda(lambda img: adaptive_equalization(img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.25257042,0.24238083,0.22564004], std=[0.28864914,0.27758299,0.25838134]),
])
    return model_patch,transform_patch