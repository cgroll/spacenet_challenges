from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


from src.data.band3_binary_mask_data import Band3BinaryMaskDataset, RandomCropImgAndLabels, ToTensorImgAndLabels, ToTensorImgAndLabelsExtension
from src.path import ProjPaths
from src.metrics import sample_logits_and_labels, logits_to_prediction, classification_cases, prediction_metrics
# %%


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



    # %% analyse / visualize all metrics

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
