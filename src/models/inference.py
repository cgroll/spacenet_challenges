from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


from src.data.band3_binary_mask_data import Band3BinaryMaskDataset, RandomCropImgAndLabels, ToTensorImgAndLabels, ToTensorImgAndLabelsExtension
from src.path import ProjPaths
# %%

def tensor_to_numpy(batch_tensor):
    return batch_tensor.cpu().detach().numpy()[0, 0, :, :]

def logits_to_prediction(logits, pred_threshold=0.5):
    pred = (logits > pred_threshold) * 1
    return pred

def sample_logits_and_labels(sample, model, DEVICE):

    data = sample['image']
    data = data[None, :] # add dimension to make it a batch

    target = sample['labels']
    target = target[None, :] # add dimension to make it a batch

    data, target = data.to(DEVICE), target.to(DEVICE)
    output = model(data)

    labels = tensor_to_numpy(target)
    logits = tensor_to_numpy(output)

    return labels, logits

def classification_cases(labels, pred):

    true_pos = (labels == 1) & (pred == 1)
    true_neg = (labels == 0) & (pred == 0)
    false_pos = (labels == 0) & (pred == 1)
    false_neg = (labels == 1) & (pred == 0)

    # check that all pixels are in one of the categories
    n_pixels = np.prod(labels.shape)
    assert true_pos.sum() + true_neg.sum() + false_neg.sum() + false_pos.sum() == n_pixels

    return true_pos, true_neg, false_pos, false_neg

def prediction_metrics(true_pos, true_neg, false_pos, false_neg):
    metrics = {'n_pixels': np.prod(true_pos.shape), 
            'true_pos': true_pos.sum(), 'true_neg': true_neg.sum(),
            'false_pos': false_pos.sum(), 'false_neg': false_neg.sum()}

    metrics_df = pd.DataFrame.from_dict({0: metrics}, orient='index')
    metrics_df['n_building'] = metrics_df['true_pos'] + metrics_df['false_neg']
    metrics_df['building_cover'] = metrics_df['n_building'] / metrics_df['n_pixels']
    metrics_df['n_union'] = metrics_df['true_pos'] + metrics_df['false_pos'] + metrics_df['false_neg']
    metrics_df['jaccard'] = metrics_df['true_pos'] / metrics_df['n_union'] # TODO: case where union is 0
    metrics_df['dice'] = 2*metrics_df['true_pos'] / (2*metrics_df['true_pos'] + metrics_df['false_pos'] + metrics_df['false_neg']) # TODO: case without any building / building prediction
    metrics_df['accuracy'] = (metrics_df['true_pos'] + metrics_df['true_neg']) / metrics_df['n_pixels']

    return metrics_df

# %%

if __name__ == '__main__':

    # %% create data loader
    
    train_path = ProjPaths.interim_sn1_data_path / "train"
    val_path = ProjPaths.interim_sn1_data_path / "val"
    test_path = ProjPaths.interim_sn1_data_path / "test"
    
    # train_dataset = Band3BinaryMaskDataset(train_path, transform=transforms.Compose([
    #                                            RandomCropImgAndLabels(384),
    #                                            ToTensorImgAndLabels()
    #                                        ]))
    # val_dataset = Band3BinaryMaskDataset(val_path, transform=transforms.Compose([
    #                                            RandomCropImgAndLabels(384),
    #                                            ToTensorImgAndLabels()
    #                                        ]))
    test_dataset = Band3BinaryMaskDataset(test_path, transform=transforms.Compose([
                                               RandomCropImgAndLabels(384),
                                               ToTensorImgAndLabels()
                                           ]))
    
    #train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
    #val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)
    #train_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    
    # %% check a data sample
    
    show_sample = False
    if show_sample:
        sample = test_dataset[237]
        
        plt.subplot(1,2,1)
        plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
        plt.subplot(1,2,2)
        plt.imshow(sample["labels"].squeeze())  # for visualization we have to remove 3rd dimension of mask
        plt.show()    
        
    # %% load model
    
    from src.models.unet_ptl import UNet
    from src.models.unet_finetune import UNetFineTune
    from pathlib import PureWindowsPath
    import torch

    model = UNetFineTune()
    chkpt_path = str(PureWindowsPath("C:\\Users\\cgrol\\OneDrive\\Dokumente\\GitHub\\spacenet_challenges\\tb_logs_2\\UNet-Fine-Tuned\\version_8\\checkpoints\\best_model-unet-epoch=03-val_loss=0.09.ckpt"))
    model = UNetFineTune.load_from_checkpoint(chkpt_path)
        
    model.eval()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    print(f'Memory usage from model: {torch.cuda.memory_allocated()}')
    
    #summary(model, input_size = (3, 384, 384), batch_size = -1)
    

    # %% inference on examples

    n_test_images = len(test_dataset)

    import random

    sample_list = random.sample(range(0, n_test_images), 30)

    # %%

    from tqdm import tqdm
    
    all_metrics_list = []

    for this_sample_id in tqdm(range(0, len(test_dataset))):

        sample = test_dataset[this_sample_id]
        labels, logits = sample_logits_and_labels(sample, model, DEVICE)
        pred = logits_to_prediction(logits, pred_threshold=0.5)
        true_pos, true_neg, false_pos, false_neg = classification_cases(labels, pred)
        metrics_df = prediction_metrics(true_pos, true_neg, false_pos, false_neg)
        metrics_df.index = [this_sample_id]

        all_metrics_list.append(metrics_df)

    all_metrics = pd.concat(all_metrics_list)

    # fix NaN metrics for images without buildings
    xx_inds_zero_build = (all_metrics['true_pos'] == 0) & (all_metrics['false_pos'] == 0)
    all_metrics.loc[xx_inds_zero_build, 'jaccard']  = 1
    all_metrics.loc[xx_inds_zero_build, 'dice']  = 1


    # %% analyse / visualize all metrics

    #sns.scatterplot(x='jaccard', y='dice', data=all_metrics)
    
    sns.scatterplot(x='jaccard', y='accuracy', data=all_metrics)
    # this_sample_id = 122

    # %%

    sns.histplot(all_metrics['building_cover'])

    # %%

    sns.scatterplot(x='building_cover', y='accuracy', data=all_metrics)

    # %%

    sns.scatterplot(x='building_cover', y='jaccard', data=all_metrics)

    # %% visualize single sample

    this_sample_id = 678
    this_sample_id = 433
    this_sample_id = 679
    this_sample_id = 677
    this_sample_id = 678
    this_sample_id = 237
    this_sample_id = 140
    this_sample_id = 4
    this_sample_id = 916
    this_sample_id = 428 # 79, 283
    # this_sample_id = 122

    print(all_metrics.loc[this_sample_id, :])

    sample = test_dataset[this_sample_id]
    labels, logits = sample_logits_and_labels(sample, model, DEVICE)
    pred = logits_to_prediction(logits, pred_threshold=0.5)
    true_pos, true_neg, false_pos, false_neg = classification_cases(labels, pred)

    classes = np.zeros(true_pos.shape)
    classes[true_neg] = 0
    classes[false_pos] = 1
    classes[false_neg] = 2
    classes[true_pos] = 3
    classes_masked = np.ma.masked_where(classes == 0, classes)
    
    prg = ['#a503fc','#f50d05', '#52f705'] # purple, red, green
    my_cmap = ListedColormap(sns.color_palette(prg).as_hex())

    fig = plt.figure(figsize=(20,40))

    plt.subplot(1,2,1)
    plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC

    plt.subplot(1,2,2)
    plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.imshow(classes_masked, alpha=0.5, cmap=my_cmap, vmin=1, vmax=3)

    img_path = ProjPaths.reports_path / 'figures'/ 'unet' / f'pred_{this_sample_id}_overlay.png'
    # plt.savefig(img_path)


    # %%
    fig = plt.figure(figsize=(10,10))

    # plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    ground_truth = sample["labels"].squeeze()
    # ground_truth_masked = np.ma.masked_where(ground_truth == 0, ground_truth)
    plt.imshow(ground_truth, alpha=0.3, cmap='jet', vmin=0, vmax=1)
    

# %%
