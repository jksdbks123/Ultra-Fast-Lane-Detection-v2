
from Utils import *

def main(RumbleStripIndicator_path,out_folder_path):
    shpfile_names = os.listdir(RumbleStripIndicator_path)
    # only keep the shp files
    shpfile_names = [name for name in shpfile_names if name.endswith('.shp')]
    shpfile_paths = [os.path.join(RumbleStripIndicator_path,name) for name in shpfile_names]
    # combine all the shp files
    combined_shp = []
    for path in shpfile_paths:
        shp = geopandas.read_file(path)
        combined_shp.append(shp)
    combined_shp = pd.concat(combined_shp)
    combined_shp = combined_shp.reset_index(drop=True)
    # save the combined shp file
    combined_shp.to_file(os.path.join(out_folder_path,'RumbleStripIndicators.shp'))
if __name__ == "__main__":
    RumbleStripIndicator_path = '../../Roadviewer/RumbleStripResults/RumbleStripLayers'
    out_folder_path = '../../Roadviewer/RumbleStripResults/'
    main(RumbleStripIndicator_path,out_folder_path)