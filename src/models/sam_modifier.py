import torch
import numpy as np
from skimage import measure
import pandas as pd
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from datetime import datetime
from tqdm import tqdm

from src.path import ProjPaths
from src.data.band3_binary_mask_data import Band3BinaryMaskDataset, RandomCropImgAndLabels, ToTensorImgAndLabels, UpperLeftCropImgAndLabels
from src.models.unet_ptl import UNet
from torchvision import transforms
from src.metrics import logits_to_prediction, sample_logits_and_labels
from src.metrics import classification_cases, prediction_metrics

def sample_to_sam_format(sample):
    img_vals = sample['image'].cpu().detach().numpy() # to numpy
    img_vals = np.moveaxis(img_vals, 0, -1) # change dimensions to HWC
    img_vals_255 = np.round(img_vals*255, 0) # colors to 0-255 range
    img_vals_255_uint =  img_vals_255.astype(np.uint8) # int, not float
    
    return img_vals_255_uint

def binary_segmentation_to_clusters(pred):
    """
    pred: matrix with zeros and ones
    """

    model_clusters = measure.label(pred)

    return model_clusters

def get_cluster_areas(model_clusters):

    properties = measure.regionprops(model_clusters)
    prop_areas = [prop.area for prop in properties]
    cluster_areas = pd.DataFrame(prop_areas, columns=['area'], index=range(1, np.max(model_clusters)+1))
    cluster_areas = cluster_areas.sort_values('area')
    
    return cluster_areas

def sam_masks_to_cluster_id_matrix(sam_masks):

    sam_clusters = np.zeros(sam_masks[0]['segmentation'].shape)
    for ii in range(0, len(sam_masks)):
        
        this_mask = sam_masks[ii]['segmentation']
        sam_clusters[this_mask] = ii+1
    # Note: 0 values correspond to non-existing SAM cluster

    return sam_clusters

def modify_cluster(this_model_cluster, sam_clusters, overlap_threshold=0.4, veto_threshold=0.05):
    cluster_size = np.sum(this_model_cluster)

    # compute intersections with SAM clusters
    intersection = this_model_cluster * sam_clusters

    # find relevant SAM clusters
    intersect_cluster_ids = np.unique(intersection)
    intersect_cluster_ids = [ii for ii in intersect_cluster_ids if ii > 0]

    # compute overlap metrics
    sam_cluster_areas = []
    intersection_areas = []

    all_intersect_clusters = np.zeros(model_clusters.shape)
    for ii in range(0, len(intersect_cluster_ids)):
        this_id = intersect_cluster_ids[ii]

        this_mask = sam_clusters == this_id
        all_intersect_clusters[this_mask] = ii+1

        this_sam_cluster_area = np.sum(this_mask)
        sam_cluster_areas.append(this_sam_cluster_area)

        intersection_mask = intersection == this_id
        this_intersection_area = np.sum(intersection_mask)
        intersection_areas.append(this_intersection_area)

    intersection_metrics = pd.DataFrame({'cluster_id': intersect_cluster_ids, 'cluster_area': sam_cluster_areas, 'intersection_area': intersection_areas})
    intersection_metrics['target_size'] = cluster_size
    intersection_metrics['overlap_ratio'] = intersection_metrics['intersection_area'] / intersection_metrics['cluster_area']

    # compute modification areas
    overlap_clusters = np.zeros(sam_clusters.shape)
    veto_clusters = np.zeros(sam_clusters.shape)
    full_overlap_clusters = np.zeros(sam_clusters.shape)

    for idx, row in intersection_metrics.iterrows():

        this_cluster_id = row['cluster_id']
        xx_inds = sam_clusters == this_cluster_id

        if row['overlap_ratio'] > overlap_threshold:

            overlap_clusters[xx_inds] = this_cluster_id

        if row['overlap_ratio'] < veto_threshold:

            veto_clusters[xx_inds] = 1
            
        if row['target_size'] == row['intersection_area']:
            
            full_overlap_clusters[xx_inds] = 1
            
    modified_cluster = this_model_cluster.copy()
    modified_cluster[veto_clusters == 1] = 0
    modified_cluster[overlap_clusters > 0] = 1
    # modified_cluster[full_overlap_clusters > 0] = 1 # optional; not really tested yet

    return modified_cluster

def get_modified_clusters(model_clusters, sam_clusters, min_pixel_size=50):

        cluster_areas = get_cluster_areas(model_clusters)
    
        all_modified_clusters = np.zeros(model_clusters.shape)
        for ii in range(0, cluster_areas.shape[0]):

            min_pixel_size = 0
        
            this_cluster_id = cluster_areas.index[ii]
            this_area = cluster_areas.iloc[ii].squeeze()
            this_model_cluster = (model_clusters == this_cluster_id)*1
            
            if this_area >= min_pixel_size:
                
                this_modified_cluster = modify_cluster(this_model_cluster, sam_clusters, overlap_threshold=0.4, veto_threshold=0.05)
                
                xx_inds = this_modified_cluster > 0
                all_modified_clusters[xx_inds] = 1

            else:
                xx_inds = this_model_cluster > 0
                all_modified_clusters[xx_inds] = 1

        return all_modified_clusters


if __name__ == "__main__":

    start_time = datetime.now().strftime('%H:%M:%S')
    print(f'Start time: {start_time}')

    # define dataset
    val_path = ProjPaths.interim_sn1_data_path / "val"
    val_dataset = Band3BinaryMaskDataset(val_path, transform=transforms.Compose([
                                            RandomCropImgAndLabels(384),
                                            ToTensorImgAndLabels()
                                        ]))
    
    test_path = ProjPaths.interim_sn1_data_path / "test"
    test_dataset = Band3BinaryMaskDataset(test_path, transform=transforms.Compose([
                                           UpperLeftCropImgAndLabels(384),
                                           ToTensorImgAndLabels()
                                       ]))

    # define segmentation model
    chkpt_path = ProjPaths.model_path / 'unet' / 'unet_ptl_v5' / 'checkpoints' / 'best_model-unet-epoch=15-val_loss=0.09.ckpt'
    model = UNet.load_from_checkpoint(chkpt_path)

    model.eval()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = 'cpu'
    model = model.to(DEVICE)

    # mem_allocated = torch.cuda.memory_allocated()
    # print(f'Memory allocated after UNet loading: {mem_allocated}')

    # set up SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    checkpoint_path = ProjPaths.model_path / 'sam' / sam_checkpoint
    model_type = "vit_h"

    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    outpath = ProjPaths.metrics_path / 'unet_ptl_v5_sam_modified.csv'

    existing_metrics = pd.read_csv(outpath, index_col='sample_id')

    all_sample_metrics = []

    for this_sample_id in tqdm(range(0, len(test_dataset))):

        if this_sample_id in existing_metrics.index:

            print(f'Skipping already computed sample: {this_sample_id}')

        else:

            # pick sample
            sample = test_dataset[this_sample_id]

            # model inference
            labels, logits = sample_logits_and_labels(sample, model, DEVICE)
            pred = logits_to_prediction(logits, pred_threshold=0.5)

            true_pos, true_neg, false_pos, false_neg = classification_cases(labels, pred)
            metrics_df = prediction_metrics(true_pos, true_neg, false_pos, false_neg)
            metrics_df.index = ['unet']

            if np.sum(pred) == 0:

                print(f'Skipping SAM modification because not buildings are detected: {this_sample_id}')
                pred_metrics_modified = metrics_df.copy()
                pred_metrics_modified.index = ['sam_modified']

            else:

                # transform binary model segmentation to cluster masks 
                model_clusters = binary_segmentation_to_clusters(pred)
                cluster_areas = get_cluster_areas(model_clusters)

                # SAM inference
                sam_image = sample_to_sam_format(sample)
                predictor.set_image(sam_image) # image embeddings
                mask_generator = SamAutomaticMaskGenerator(sam) # define algo parameters
                sam_masks = mask_generator.generate(sam_image) # apply segmentation algo
                sam_clusters = sam_masks_to_cluster_id_matrix(sam_masks)
                
                # apply SAM modification to all clusters
                all_modified_clusters = get_modified_clusters(model_clusters, sam_clusters)
                pred_modified = (all_modified_clusters > 0)*1

                # compute metrics
                labels = sample['labels']
                labels = labels.cpu().detach().numpy()[0, :, :]

                true_pos, true_neg, false_pos, false_neg = classification_cases(labels, all_modified_clusters)
                pred_metrics_modified = prediction_metrics(true_pos, true_neg, false_pos, false_neg)
                pred_metrics_modified.index = ['sam_modified']

            metrics_comparison = pd.concat([metrics_df, pred_metrics_modified], axis=0)
            metrics_comparison

            print(metrics_comparison)

            pred_metrics_modified.index = [this_sample_id]
            all_sample_metrics.append(pred_metrics_modified)

            all_metrics = pd.concat(all_sample_metrics)

            # fix NaN metrics for images without buildings
            xx_inds_zero_build = (all_metrics['true_pos'] == 0) & (all_metrics['false_pos'] == 0)
            all_metrics.loc[xx_inds_zero_build, 'jaccard']  = 1
            all_metrics.loc[xx_inds_zero_build, 'dice']  = 1
            all_metrics = all_metrics.reset_index()
            all_metrics.rename({'index': 'sample_id'}, axis=1, inplace=True)
            all_metrics.to_csv(outpath, index=False)

            end_time = datetime.now().strftime('%H:%M:%S')
            print(f'End time: {end_time}')

        