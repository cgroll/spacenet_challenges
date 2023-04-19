#%%
# import segmentation_models_pytorch as smp
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
    
    train_dataset = Band3BinaryMaskDataset(train_path, transform=transforms.Compose([
                                               RandomCropImgAndLabels(384),
                                               ToTensorImgAndLabels()
                                           ]))
    val_dataset = Band3BinaryMaskDataset(val_path, transform=transforms.Compose([
                                               RandomCropImgAndLabels(384),
                                               ToTensorImgAndLabels()
                                           ]))
    
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
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

    # %% clear GPU memory - probably not required

    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print(f'Memory usage after cache empty and gc collect: {torch.cuda.memory_allocated()}')

    # %%

    def save_model(this_epoch, model, optimizer, fpath):
        """
        Function to save the trained model to disk.
        TODO: add loss value?
        """
        torch.save({
                    'epoch': this_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, fpath)

    # %% define model
    
    from src.models.unet import UNet
    from torchsummary import summary
    # device = torch.device("cuda")

    model = UNet(out_classes=1)

    for param in model.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)
    print(f'Memory usage from model: {torch.cuda.memory_allocated()}')
    
    #summary(model, input_size = (3, 384, 384), batch_size = -1)

    # %%

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.00008),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # %%

    import datetime
    now = datetime.datetime.now()
    
    
    # %%
    model.train()

    n_epochs = 10
    running_best_loss = 100000000
    
    for this_epoch in range(0, n_epochs):
        this_epoch_id_print = this_epoch + 1

        total_train_loss = 0
        total_test_loss = 0

        for batch_idx, this_batch in enumerate(train_dataloader):
            data = this_batch['image']
            target = this_batch['labels']
            target_ext = target
            # target_ext = target[None, :]
            
            data, target_ext = data.to(DEVICE), target_ext.to(DEVICE)
            # target_ext.requires_grad = True
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target_ext)
            
            total_train_loss += loss

            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            #del loss
            #del output
            #torch.cuda.empty_cache()
            #gc.collect()
            
            if batch_idx % 200 == 0:
                now = datetime.datetime.now()
                this_time_str = now.strftime('%H:%M:%S')

                print('{}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    this_time_str, this_epoch_id_print, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss_val))
                # print(f'Memory usage: {torch.cuda.memory_allocated()}')

        
        with torch.no_grad():

            model.eval()

            for batch_idx, this_batch in enumerate(val_dataloader):

                data = this_batch['image']
                target = this_batch['labels']
                target_ext = target
                
                data, target_ext = data.to(DEVICE), target_ext.to(DEVICE)

                output = model(data)
                loss = loss_fn(output, target_ext)

                total_test_loss += loss

        
        avg_train_loss = total_train_loss / len(train_dataloader.dataset) 
        avg_test_loss = total_test_loss / len(val_dataloader.dataset)

        if avg_test_loss < running_best_loss:
            running_best_loss = avg_test_loss

            fpath = ProjPaths.unet_path / 'unet_v1_best.pth'
            save_model(this_epoch, model, optimizer, fpath)
            print(f'Saved new best model in EPOCH {this_epoch_id_print}')

        print("[INFO] EPOCH: {}/{}".format(this_epoch_id_print, n_epochs))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(avg_train_loss, avg_test_loss))


    # %% save final model
    
    fpath = ProjPaths.unet_path / 'unet_v1_final.pth'
    save_model(this_epoch, model, optimizer, fpath)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # %%
    
    ##### BELOW: Experiment with pre-computed Unet
    
    
    # %%
    
#     model = smp.Unet(
#         encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#         in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#         classes=1,                      # model output channels (number of classes in your dataset)
#     )

#     for param in model.parameters():
#         param.requires_grad = True
        
#     model.train()
        
#     # %%
#     from torchmetrics import JaccardIndex
    
#     criterion = JaccardIndex(task="binary")
    
#     # %%
    
#     import torch.optim as optim
    
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
#     # %%
    
#     for epoch in range(2):  # loop over the dataset multiple times

#         running_loss = 0.0
#         for i, data in enumerate(train_dataloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs = data['image']
#             labels = data['labels']
#             labels = labels.reshape(1, 1, 384, 384)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = model(inputs.float)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#                 running_loss = 0.0

#     print('Finished Training')
# # %%
