# %%
from sklearn.model_selection import train_test_split
import pandas as pd
from src.path import ProjPaths
import os
from os import listdir
from os.path import isfile, join
import shutil

from src.file_tools import get_all_files_in_path, extract_img_id_from_str
from src.file_tools import derive_8band_file_names_from_image_ids, derive_json_file_names_from_image_ids

# %%

def copy_files_to_dst_folder(file_list_df, src_data_path, dataset_type_folder):

    file_types = ['3band', '8band', 'geojson']        
    for this_file_type in file_types:
        
        src_path_folder = src_data_path / this_file_type
        dst_path_folder = dataset_type_folder / this_file_type
        
        # remove existing directory
        if os.path.exists(dst_path_folder):
            shutil.rmtree(dst_path_folder)
            
        dst_path_folder.mkdir(parents=True, exist_ok=False)
        
        for this_file in file_list_df[this_file_type]:
            src_path = os.path.join(src_path_folder, this_file)
            dst_path = os.path.join(dst_path_folder, this_file)
            
            shutil.copyfile(src_path, dst_path)
        

# %%


if __name__ == '__main__':
    
    # %% CONFIG
    
    RANDOM_STATE = 42
    
    TEST_SIZE = 0.15
    VAL_SIZE_FROM_TRAIN = 0.15
    
    # %%
    
    data_path = ProjPaths().raw_sn1_data_path / "train"
    
    path_3band = data_path / "3band"
    path_8band = data_path / "8band"
    path_geojson = data_path / "geojson"
    
    # get all files
    files_3band = get_all_files_in_path(path_3band)
    files_8band = get_all_files_in_path(path_8band)
    files_json = get_all_files_in_path(path_geojson)
    
    # %%
    
    # get image IDs from 3band files
    img_ids = [extract_img_id_from_str(fname) for fname in files_3band]
    
    # derive 8band and geojson files names from image IDs
    derived_8band_files = derive_8band_file_names_from_image_ids(img_ids)
    derived_json_files = derive_json_file_names_from_image_ids(img_ids)

    # %% sanity checks
    assert len(files_3band) == len(files_8band)
    assert len(files_3band) == len(files_json)
    
    # %%
    
    fname_df = pd.DataFrame({'3band': files_3band, '8band': derived_8band_files})
    fname_df.head(3)
    
    # %% split into train_val and test
    
    # get unique image IDs
    X_train_val, X_test, y_train_val, y_test = train_test_split(fname_df, derived_json_files,
                                                        test_size=TEST_SIZE, 
                                                        random_state=RANDOM_STATE)
    X_train_val['geojson'] = y_train_val
    X_test['geojson'] = y_test

    # %% split into train and val
    
    # get unique image IDs
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                        test_size=VAL_SIZE_FROM_TRAIN, 
                                                        random_state=RANDOM_STATE)
    X_train['geojson'] = y_train
    X_val['geojson'] = y_val
    
    # %%
    
    X_test

    # %% 
    
    src_data_path = ProjPaths().raw_sn1_data_path / "train" # for all datasets: train, test, val, train_val
    
    
    path_test = ProjPaths.interim_sn1_data_path / "test"
    path_train_val = ProjPaths.interim_sn1_data_path / "train_val"
    path_train = ProjPaths.interim_sn1_data_path / "train"
    path_val = ProjPaths.interim_sn1_data_path / "val"
    
    copy_files_to_dst_folder(X_test, src_data_path, path_test)
    copy_files_to_dst_folder(X_train_val, src_data_path, path_train_val)
    copy_files_to_dst_folder(X_train, src_data_path, path_train)
    copy_files_to_dst_folder(X_val, src_data_path, path_val)

