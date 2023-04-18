# %%
import geopandas as gpd
import rioxarray
from rasterio import features
from PIL import Image
import numpy as np
from numpy import asarray
from rasterio.plot import show
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import shutil

from src.path import ProjPaths
from src.file_tools import get_all_files_in_path, extract_img_id_from_str


def rasterize_single_image(img_src, geo_src, raster_out_path):

    img_darr = rioxarray.open_rasterio(img_src)
    geo_df = gpd.read_file(geo_src)
    
    if geo_df.shape[0] == 0:
        rasterized = np.zeros(img_darr.rio.shape, dtype='uint8')
        
    else:
    
        rasterized = features.rasterize(geo_df.geometry.values,
                                    out_shape=img_darr.rio.shape,
                                    fill=0,
                                    transform=img_darr.rio.transform(),
                                    all_touched=False,
                                    default_value=1,
                                    dtype=None
                                    )
    
    # write to disk
    im = Image.fromarray(rasterized)
    im.save(raster_out_path)
    

# %%

if __name__ == '__main__':
    
    # %% config

    # band_file_types = ['3band', '8band']
    # Note: only implemented for 3band yet

    data_types = ['train', 'val', 'train_val', 'test']
    
    for this_data_type in data_types:
        this_data_type_path = ProjPaths.interim_sn1_data_path / this_data_type
        
        img_path = this_data_type_path / '3band'
        json_path = this_data_type_path / 'geojson'
        
        out_path = this_data_type_path / 'geo_raster_3band'
        
        # remove existing directory
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True, exist_ok=False)
        
        # get list of files
        img_file_list = get_all_files_in_path(img_path)
        
        # get image IDs
        img_ids = [extract_img_id_from_str(fname) for fname in img_file_list]
        
        # derive json file names
        derived_json_files = [('Geo_' + this_img_id + '.geojson') for this_img_id in img_ids]
        
        # derive geo-raster file names
        derived_georaster_files = [('GeoRaster_' + this_img_id + '.png') for this_img_id in img_ids]
        
        for this_img, this_json, this_georaster in tqdm(zip(img_file_list, derived_json_files, derived_georaster_files)):
            
            fpath_img = img_path / this_img
            fpath_json = json_path / this_json
            fpath_output = out_path / this_georaster
            
            rasterize_single_image(fpath_img, fpath_json, fpath_output)
            

    # for each folder
    # - get list of files
    # - derive image IDs
    # - derive geojson files
    # - load .tif
    # - load .json
    # - rasterize shapes

    # %% test case
    
    # fpath_img = '/home/chris/research/spacenet_challenges/data/interim/SN1/test/3band/3band_AOI_1_RIO_img6899.tif'
    # fpath_json = '/home/chris/research/spacenet_challenges/data/interim/SN1/test/geojson/Geo_AOI_1_RIO_img6899.geojson'
    # fpath_output = '/home/chris/research/spacenet_challenges/data/interim/SN1/test/geo_raster_3band/GeoRaster_AOI_1_RIO_img6899.png'
    
    # rasterize_single_image(fpath_img, fpath_json, fpath_output)
    
    # %% test: load back again and visualize
    
    # rasterized_reloaded = Image.open(fpath_output)
    # rasterized_reloaded_np = asarray(rasterized_reloaded)
    # np.min(rasterized_reloaded_np)
    # np.max(rasterized_reloaded_np)
    
    # show(rasterized)
    # show(rasterized_reloaded_np)
    
