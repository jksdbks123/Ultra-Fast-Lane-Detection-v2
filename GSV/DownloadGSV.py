import requests
import json
import numpy as np
import geopandas
import os
from tqdm import tqdm
from shapely.geometry import Point
import time


def get_heading(x,y):
    heading = np.arctan2(x,y)*180/np.pi
    if heading < 0:
        heading += 360
    return heading
def get_image(panoID,heading,fov,img_save_path,key):
    params = {
      'pano': panoID,
      'size': '640x640', # maximum 640x640 pixels
      'heading': '{}'.format(heading), # 90 East; 180 South
      'fov':'{}'.format(fov),
      # 'key':'AIzaSyA79jbqreAsWqQa5up0B4XI2LFd8RDAFoA'
        'key':key
    }
    img_reqs = 'https://maps.googleapis.com/maps/api/streetview?pano={pano}&size={size}&heading={heading}&fov={fov}&key={key}'
    img_reqs = img_reqs.format(pano = params['pano'],key = params['key'],size = params['size'],heading = params['heading'],fov = params['fov'])
    response = requests.get(img_reqs)
    if response.status_code != 200:
        return False
    with open(img_save_path, 'wb') as file:
        file.write(response.content)
    response.close()
    return True

def get_streetview_panoID(lon,lat,heading,key):
    params = {
      'size': '640x640', # maximum 640x640 pixels
      'location': '{},{}'.format(lat,lon),
      'heading': '{}'.format(heading), # 90 East; 180 South
      'fov':'120',
      'return_error_code':'true',

      'radius':'10',
      'source':'outdoor',
       'key':key
    }
    meta_reqs = 'https://maps.googleapis.com/maps/api/streetview/metadata?size={size}&location={location}&heading={heading}&source={source}&radius={radius}&key={key}'
    meta_reqs = meta_reqs.format(size = params['size'],location = params['location'],heading = params['heading'],source = params['source'],radius = params['radius'],key = params['key'])
    response = requests.get(meta_reqs)
    MetaData = response.json()
    response.close()
    return MetaData

def download_metafiles(selected_routes_26911,GSV_save_path,gsv_sampling_interval,key):
    if not os.path.exists(GSV_save_path):
        os.makedirs(GSV_save_path)
    Meta_Folder = os.path.join(GSV_save_path,'MetaData')
    if not os.path.exists(Meta_Folder):
        os.makedirs(Meta_Folder)
    img_Folder = os.path.join(GSV_save_path,'Images')
    if not os.path.exists(img_Folder):
        os.makedirs(img_Folder)

    
    for i,row in selected_routes_26911.iterrows():
        route_folder_meta = os.path.join(Meta_Folder,f'{row.RouteID}')
        if not os.path.exists(route_folder_meta):
            os.makedirs(route_folder_meta)
        
        route_folder_img = os.path.join(img_Folder,f'{row.RouteID}')
        if not os.path.exists(route_folder_img):
            os.makedirs(route_folder_img)
            
        route_len = row.geometry.length
        sample_num = int(route_len / gsv_sampling_interval) + 1
        route_sampled_points = []
        headings = []
        for j in range(sample_num):
            
            lon_cur,lat_cur = row.geometry.interpolate(j * gsv_sampling_interval).xy
            if gsv_sampling_interval * j + 10 > route_len:
                lon_next,lat_next = row.geometry.interpolate(route_len).xy
            else:
                lon_next,lat_next = row.geometry.interpolate(j * gsv_sampling_interval + 10).xy
            
            vec = np.array([lon_next[0] - lon_cur[0],lat_next[0] - lat_cur[0]])
            mod = np.sqrt(np.sum(vec**2))
            vec_unit = vec/mod
            heading = get_heading(vec_unit[0],vec_unit[1])
            headings.append(heading)
            route_sampled_points.append(Point(lon_cur[0],lat_cur[0]))

        route_sampled_points = geopandas.GeoSeries(route_sampled_points,crs='epsg:26911')
        route_sampled_points = route_sampled_points.to_crs(epsg = '4326')
        

        Metas = []
        panos = []
        print(f'Route {row.RouteID} has {len(route_sampled_points)} images')
        for j in tqdm(range(len(route_sampled_points))):
            lon_cur,lat_cur = route_sampled_points[j].xy
            Meta = get_streetview_panoID(lon_cur[0],lat_cur[0],headings[j],key)
            if Meta['status'] != 'OK':
                continue
            pano = Meta['pano_id']
            if pano in panos:
                continue
            Meta['heading'] = headings[j]
            panos.append(pano)
            Metas.append(Meta)
        print(f'Route {row.RouteID} has {len(Metas)} acutal images, now saving metas')
        # save Metas to json
        for j,Meta in enumerate(Metas):
            meta_path = os.path.join(route_folder_meta,f'{panos[j]}.json')
            with open(meta_path,'w') as f:
                json.dump(Meta,f)

def download_images(GSV_save_path,key,fov = '120'):

    meta_folders = os.listdir(os.path.join(GSV_save_path,'MetaData'))
    ActualAPIreqs = 0
    total_imgs_needed = 0
    for meta_folder in meta_folders:
        meta_folder_path = os.path.join(GSV_save_path,'MetaData',meta_folder)
        meta_files = os.listdir(meta_folder_path)
        for meta_file in meta_files:
            img_save_path = os.path.join(GSV_save_path,'Images',meta_folder,f'{meta_file[:-5]}.jpg')
            if not os.path.exists(img_save_path):
                total_imgs_needed += 1
    print(f'Total images needed: {total_imgs_needed}')

    img_folder = os.path.join(GSV_save_path,'Images')
    for meta_folder in meta_folders:
        meta_folder_path = os.path.join(GSV_save_path,'MetaData',meta_folder)
        meta_files = os.listdir(meta_folder_path)
        for meta_file in meta_files:
            with open(os.path.join(meta_folder_path,meta_file),'r') as f:
                meta = json.load(f)
            pano = meta['pano_id']
            heading = meta['heading']
            img_save_path = os.path.join(img_folder,meta_folder,f'{meta_file[:-5]}.jpg')
            if os.path.exists(img_save_path):
                continue         
            while True:   
                flag = get_image(pano,heading,fov,img_save_path,key)
                if flag:
                    ActualAPIreqs += 1
                    break
                else:
                    print('Key Error Happened, wait for 10 sec and retry...')
                    time.sleep(10)
            if ActualAPIreqs%200 == 0:
                print(f'Actual API requests: {ActualAPIreqs}')

def main(selected_routes_26911,GSV_save_path,key,gsv_sampling_interval = 304.87):
    print('Start downloading metafiles...')
    download_metafiles(selected_routes_26911,GSV_save_path,gsv_sampling_interval,key)
    print('Start downloading images...')
    download_images(GSV_save_path,key)

    

if __name__ == '__main__':
    FS_system = geopandas.read_file('../../Roadviewer/GeneralGIS/FSystem/FSystem.shp')
    selected_routes_26911 = FS_system.loc[FS_system.FSystem.isin(['0','1','2','3','4','5'])]
    selected_routes_26911 = selected_routes_26911.to_crs(epsg = '26911')
    GSV_save_path = r'../../Roadviewer/GSVDownload'
    key = 'AIzaSyA79jbqreAsWqQa5up0B4XI2LFd8RDAFoA'

    main(selected_routes_26911,GSV_save_path,key)