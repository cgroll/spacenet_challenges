#%%
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from src.data.band3_binary_mask_data import Band3BinaryMaskDataset, RandomCropImgAndLabels, ToTensorImgAndLabels, ToTensorImgAndLabelsExtension
from src.path import ProjPaths
#%%

if __name__ == '__main__':

    # %% create data loader
    
    train_path = ProjPaths.interim_sn1_data_path / "train"
    val_path = ProjPaths.interim_sn1_data_path / "val"
    test_path = ProjPaths.interim_sn1_data_path / "test"
    
    train_dataset = Band3BinaryMaskDataset(train_path, transform=transforms.Compose([
                                               RandomCropImgAndLabels(384),
                                               ToTensorImgAndLabels()
                                           ]))
    val_dataset = Band3BinaryMaskDataset(val_path, transform=transforms.Compose([
                                               RandomCropImgAndLabels(384),
                                               ToTensorImgAndLabels()
                                           ]))
    test_dataset = Band3BinaryMaskDataset(test_path, transform=transforms.Compose([
                                               RandomCropImgAndLabels(384),
                                               ToTensorImgAndLabels()
                                           ]))
    
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)
    #train_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    
    # %% check a data sample
    
    show_sample = False
    if show_sample:
        sample = train_dataset[106]
        
        plt.subplot(1,2,1)
        plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
        plt.subplot(1,2,2)
        plt.imshow(sample["labels"].squeeze())  # for visualization we have to remove 3rd dimension of mask
        plt.show()    
        
    # %%
    
    from torchmetrics import JaccardIndex

    import torch
    #from segmentation_models_pytorch.losses import JaccardLoss
    
    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
    TRAINING = True

    # Set num of epochs
    EPOCHS = 12

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function
    #loss_fn = JaccardIndex('binary')

    import torch.nn as nn
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()

    #loss_fn = JaccardLoss('binary')

    # %% define model
    
    from src.models.unet import UNet

    fpath = ProjPaths.unet_path / 'unet_v1_final.pth'

    model = UNet(out_classes=1)
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model = model.to(DEVICE)
    print(f'Memory usage from model: {torch.cuda.memory_allocated()}')
    
    #summary(model, input_size = (3, 384, 384), batch_size = -1)

    # %%

    import datetime
    now = datetime.datetime.now()
    
    
    # %%
    
    total_test_loss = 0

    with torch.no_grad():

        model.eval()

        for batch_idx, this_batch in enumerate(test_dataloader):

            data = this_batch['image']
            target = this_batch['labels']
            target_ext = target
            
            data, target_ext = data.to(DEVICE), target_ext.to(DEVICE)

            output = model(data)
            loss = loss_fn(output, target_ext)

            total_test_loss += loss

    
    avg_test_loss = total_test_loss / len(test_dataloader.dataset)

    print("Test loss: {:.4f}".format(avg_test_loss))


    # %% inference on examples

    n_test_images = len(test_dataset)

    import random

    sample_list = random.sample(range(0, n_test_images), 30)

    # %%
    
    this_sample_id = 679
    this_sample_id = 677
    this_sample_id = 678

    for this_sample_id in sample_list:
        #sample = test_dataset[677]
        #sample = test_dataset[678]
        sample = test_dataset[this_sample_id]
        data = sample['image']
        data = data[None, :]
        
        target = sample['labels']
        target_ext = target[None, :]

        data, target_ext = data.to(DEVICE), target_ext.to(DEVICE)
        output = model(data)

        pred = output.cpu().detach().numpy()
        pred_image = pred[0, 0, :, :]

        threshold = 0.5
        pred_binary = (pred_image > threshold)*1.0

        import numpy as np
        pred_binary_masked = np.ma.masked_where(pred_binary == 0, pred_binary)

        fig = plt.figure(figsize=(24,8))
        plt.subplot(1,3,1)
        plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
        plt.subplot(1,3,2)
        plt.imshow(sample["labels"].squeeze())
        plt.subplot(1,3,3)
        plt.imshow(pred_binary)

        img_path = ProjPaths.reports_path / 'figures'/ 'unet' / f'pred_{this_sample_id}.png'
        plt.savefig(img_path)

        fig = plt.figure(figsize=(16,8))

        this_cmap = 'Oranges'
        this_vmax = 2

        this_cmap = 'plasma'
        this_vmax = 1

        plt.subplot(1,2,1)
        plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
        ground_truth = sample["labels"].squeeze()
        ground_truth_masked = np.ma.masked_where(ground_truth == 0, ground_truth)
        plt.imshow(ground_truth_masked, alpha=0.3, cmap=this_cmap, vmin=0, vmax=this_vmax)
        
        plt.subplot(1,2,2)
        plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
        plt.imshow(pred_binary_masked, alpha=0.3, cmap=this_cmap, vmin=0, vmax=this_vmax)

        img_path = ProjPaths.reports_path / 'figures'/ 'unet' / f'pred_{this_sample_id}_overlay.png'
        plt.savefig(img_path)


# %%
