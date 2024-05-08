import geopandas
import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
import json
from tqdm import tqdm

def generate_point_layers(GSV_save_path,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # walk through the GSV_save_path and generate point layers
    Meta_Folder = os.path.join(GSV_save_path,'MetaData')
    if not os.path.exists(Meta_Folder):
        return
    Img_Folder = os.path.join(GSV_save_path,'Images')
    if not os.path.exists(Img_Folder):
        return
    Meta_Folders = os.listdir(Meta_Folder) # get all the folders in the Meta_Folder, each folder is a route
    for folder in tqdm(Meta_Folders):
        Meta_Folder_route = os.path.join(Meta_Folder,folder)
        Img_Folder_route = os.path.join(Img_Folder,folder)
        if not os.path.exists(Img_Folder_route):
            continue
        Meta_Files = os.listdir(Meta_Folder_route) # get all the files in the Meta_Folder_route, each file is a sample point
        metas = []
        lngs,lats = [],[]
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
        # save the metas as a point layer (save multiple dictionaries in a list as a point layer)
        if len(metas) == 0:
            continue
        metas = pd.DataFrame(metas)
        # delete the 'copyright' column
        metas = metas.drop(columns=['copyright'])
        # save the metas as a point layer

        metas = geopandas.GeoDataFrame(metas,geometry=geopandas.points_from_xy(lngs,lats))
        metas.crs = 'epsg:4326'
        metas.to_file(os.path.join(output_path,f'{folder}.shp'))
        
if __name__ == '__main__':
    GSV_save_path = '../../Roadviewer/GSVDownload'
    output_path = '../../Roadviewer/GSVPointLayers'
    generate_point_layers(GSV_save_path,output_path)