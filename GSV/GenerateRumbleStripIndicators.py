import geopandas
import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
import json
from tqdm import tqdm
from Utils import *

def generate_rumble_strip_layers(GSV_save_path,output_path,model_patch,net_lane,transform_patch,img_transforms_lane,cfg):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # walk through the GSV_save_path and generate point layers
    Meta_Folder = os.path.join(GSV_save_path,'MetaData')
    if not os.path.exists(Meta_Folder):
        return
    Img_Folder = os.path.join(GSV_save_path,'Images')
    if not os.path.exists(Img_Folder):
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    RumbleStrip_labeled_img_folder = os.path.join(output_path,'RumbleStripLabeledImages')
    RumbleStrip_indicator_folder = os.path.join(output_path,'RumbleStripIndicators')
    RumbleStrip_layer_folder = os.path.join(output_path,'RumbleStripLayers')
    if not os.path.exists(RumbleStrip_labeled_img_folder):
        os.makedirs(RumbleStrip_labeled_img_folder)
    if not os.path.exists(RumbleStrip_indicator_folder):
        os.makedirs(RumbleStrip_indicator_folder)
    if not os.path.exists(RumbleStrip_layer_folder):
        os.makedirs(RumbleStrip_layer_folder)
    
    Meta_Folders = os.listdir(Meta_Folder) # get all the folders in the Meta_Folder, each folder is a route
    for folder in tqdm(Meta_Folders):
        # folder is the route ID
        # create a folder in the output_path to save the Labeled Images
        Labeled_Img_Folder_route = os.path.join(RumbleStrip_labeled_img_folder,folder)
        if not os.path.exists(Labeled_Img_Folder_route):
            os.makedirs(Labeled_Img_Folder_route)
        # create a folder in the output_path to save the Rumble Strip Indicators
        Indicator_Folder_route = os.path.join(RumbleStrip_indicator_folder,folder)
        if not os.path.exists(Indicator_Folder_route):
            os.makedirs(Indicator_Folder_route)

        Meta_Folder_route = os.path.join(Meta_Folder,folder)
        Img_Folder_route = os.path.join(Img_Folder,folder)
        if not os.path.exists(Img_Folder_route):
            continue
        Meta_Files = os.listdir(Meta_Folder_route) # get all the files in the Meta_Folder_route, each file is a sample point
        metas = []
        lngs,lats = [],[]
        route_rumble_strip_labels = []

        for file in Meta_Files:
            Meta_File = os.path.join(Meta_Folder_route,file) # meta data is json file
            with open(Meta_File,'r') as f:
                # keys: 'date', 'location','pano_id','heading', we will save them as columns
                meta = json.load(f)
            # get the image path, also save it as a column in the point layer (absolute path)
            Img_path = os.path.join(Img_Folder_route,f'{file[:-5]}.jpg')

            if not os.path.exists(Img_path):
                continue
            meta['Img_path'] = Img_path
            meta['RouteID'] = folder
            # get the location, save it as a column in the point layer
            lngs.append(meta['location']['lng'])
            lats.append(meta['location']['lat'])
            # delete the 'location' in the meta dictionary
            del meta['location']
            # meta is a dictionary with keys: 'date', 'location','pano_id','heading','Img_path','RouteID'
            metas.append(meta)

            rumble_strip_labels,valid_lane_coords,raw_GSV = get_rumble_strip_label(Img_path,model_patch,net_lane,
                                                                          transform_patch,img_transforms_lane,cfg)
            if 1 in rumble_strip_labels:
                route_rumble_strip_labels.append(1)
            else:
                route_rumble_strip_labels.append(0)

            df,labeled_GSV = get_rumble_strip_indicator_file(rumble_strip_labels,valid_lane_coords,raw_GSV)
            # save the rumble strip indicators as a csv file to the Indicator_Folder_route
            df.to_csv(os.path.join(Indicator_Folder_route,f'{file[:-5]}.csv'),index=False)
            # save the labeled GSV to the Labeled_Img_Folder_route
            cv2.imwrite(os.path.join(Labeled_Img_Folder_route,f'{file[:-5]}.jpg'),labeled_GSV)

        # save the metas as a point layer (save multiple dictionaries in a list as a point layer)
        if len(metas) == 0:
            continue
        metas = pd.DataFrame(metas)
        # delete the 'copyright' column
        metas = metas.drop(columns=['copyright'])
        # add a column 'rumble_strip_labels' to the metas
        metas['RS_binary'] = route_rumble_strip_labels
        # save the metas as a point layer
        metas = geopandas.GeoDataFrame(metas,geometry=geopandas.points_from_xy(lngs,lats))
        metas.crs = 'epsg:4326'
        metas.to_file(os.path.join(RumbleStrip_layer_folder,f'{folder}.shp'))
        
if __name__ == '__main__':
    GSV_save_path = '../../Roadviewer/GSVDownload'
    output_path = '../../Roadviewer/RumbleStripResults'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config_file_path,ckpt_path = '../configs/culane_res34.py','../culane_res34.pth'
    net_lane,img_transforms_lane,cfg = get_lane_detector(config_file_path,ckpt_path,device)
    save_path = '../../Roadviewer/TrainingRes18'
    # read model from checkpoint
    ckpt_path = os.path.join(save_path,'resnet_custom_model_best.pth')
    model_patch,transform_patch =  get_rumble_strip_classifier(ckpt_path,device)

    generate_rumble_strip_layers(GSV_save_path,output_path,model_patch,net_lane,transform_patch,img_transforms_lane,cfg)