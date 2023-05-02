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
import cv2

from src.path import ProjPaths
from src.file_tools import get_all_files_in_path, extract_img_id_from_str
from scipy.ndimage import distance_transform_edt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch

## set up SAM
out_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
checkpoint = os.path.join(out_dir, 'sam_vit_h_4b8939.pth')

sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=15)

def get_merged_distance_transform(rasterized):
    
    im_dist = distance_transform_edt(rasterized)

    rasterized_inv = 1 - rasterized
    im_dist_inv = distance_transform_edt(rasterized_inv)

    im_dist_merged = (-1)*im_dist.copy()
    im_dist_merged[im_dist_inv > 0] = im_dist_inv[im_dist_inv > 0]
    
    return im_dist_merged

def dist_transform_single_image(img_src, geo_src, raster_out_path):

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
        
    # apply distance transformation
    im_dist_merged = get_merged_distance_transform(rasterized)

    # write to disk
    # im = Image.fromarray(im_dist_merged)
    # im.save(raster_out_path)
    
    cv2.imwrite(str(raster_out_path), im_dist_merged)
    

def segment_anything_single_image(fpath_img, fpath_output_2):

    img_darr = rioxarray.open_rasterio(fpath_img)
    
    img_vals = img_darr.values
    img_vals = np.moveaxis(img_vals, 0, -1) # change dimensions to HWC
    
    masks = mask_generator.generate(img_vals)
    
    all_mask_classes = np.zeros(img_vals.shape[:2], dtype='uint8')
    for ii in range(0, len(masks)):

        this_mask = masks[ii]['segmentation']
        all_mask_classes[this_mask] = ii+1
    
    rasterized = all_mask_classes > 0
    
    im_dist_merged = get_merged_distance_transform(rasterized)
    
    # write to disk
    # im = Image.fromarray(im_dist_merged)
    # im.save(fpath_output_2)
    
    cv2.imwrite(str(fpath_output_2), im_dist_merged)
    
    
    

# %%

if __name__ == '__main__':
    
    # %% config

    # band_file_types = ['3band', '8band']
    # Note: only implemented for 3band yet

    data_types = ['train', 'val', 'test'] # ['train', 'val', 'train_val', 'test']
    
    for this_data_type in data_types:
        this_data_type_path = ProjPaths.interim_sn1_data_path / this_data_type
        
        img_path = this_data_type_path / '3band'
        json_path = this_data_type_path / 'geojson'
        
        out_path_1 = this_data_type_path / 'sam_distance_transformed'
        out_path_2 = this_data_type_path / 'distance_transformed_label'
        
        # # remove existing directory
        # if os.path.exists(out_path_1):
        #     shutil.rmtree(out_path_1)
        # out_path_1.mkdir(parents=True, exist_ok=False)
        
        # if os.path.exists(out_path_2):
        #     shutil.rmtree(out_path_2)
        # out_path_2.mkdir(parents=True, exist_ok=False)
        
        out_path_1.mkdir(parents=True, exist_ok=True)
        out_path_2.mkdir(parents=True, exist_ok=True)
        
        # get list of files
        img_file_list = get_all_files_in_path(img_path)
        
        # get image IDs
        img_ids = [extract_img_id_from_str(fname) for fname in img_file_list]
        
        # derive json file names
        derived_json_files = [('Geo_' + this_img_id + '.geojson') for this_img_id in img_ids]
        
        # derive SAM mask and distance transform file names
        derived_dist_trans_files = [('dist_trans_' + this_img_id + '.png') for this_img_id in img_ids]
        derived_sam_files = [('sam_' + this_img_id + '.png') for this_img_id in img_ids]
        
        for this_img, this_json, this_dist_trans, this_sam in tqdm(zip(img_file_list, derived_json_files, derived_dist_trans_files, derived_sam_files)):
            
            fpath_img = img_path / this_img
            fpath_json = json_path / this_json
            fpath_output_1 = out_path_1 / this_sam
            fpath_output_2 = out_path_2 / this_dist_trans
            
            if os.path.exists(fpath_output_1):
                print(f'Skipping for output {fpath_output_1}')
                
            else:
                dist_transform_single_image(fpath_img, fpath_json, fpath_output_2)
                segment_anything_single_image(fpath_img, fpath_output_1)
            
            
            
            
            

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
    
