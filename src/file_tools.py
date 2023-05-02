import os
from os import listdir
from os.path import isfile, join


def get_all_files_in_path(this_path):
    
    dir_files = [f for f in listdir(this_path) if isfile(join(this_path, f))]
    return dir_files

def extract_img_id_from_str(filename):
    """
    filename: str of filename without path
    """
    
    filename_no_extension = filename.split('.')[0]
    img_id = filename_no_extension.split('_', 1)[1]
    
    return img_id

def derive_8band_file_names_from_image_ids(img_ids):
    return [('8band_' + this_img_id + '.tif') for this_img_id in img_ids]

def derive_json_file_names_from_image_ids(img_ids):
    return [('Geo_' + this_img_id + '.geojson') for this_img_id in img_ids]

def derive_3band_file_names_from_image_ids(img_ids):
    return [('3band_' + this_img_id + '.tif') for this_img_id in img_ids]

def derive_geo_raster_file_names_from_image_ids(img_ids):
    return [('GeoRaster_' + this_img_id + '.png') for this_img_id in img_ids]

def derive_sam_dist_transf_file_names_from_image_ids(img_ids):
    return [('sam_' + this_img_id + '.png') for this_img_id in img_ids]

def derive_label_dist_transf_file_names_from_image_ids(img_ids):
    return [('dist_trans_' + this_img_id + '.png') for this_img_id in img_ids]

