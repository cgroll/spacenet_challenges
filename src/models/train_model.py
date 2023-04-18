#%%
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from src.data.band3_binary_mask_data import Band3BinaryMaskDataset, RandomCropImgAndLabels, ToTensorImgAndLabels
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # %% check a data sample
    
    sample = train_dataset[106]
    
    plt.subplot(1,2,1)
    plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(sample["labels"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.show()    
    
    print('hello')

    # %%
    
    from src.models.unet import UNet
    
    model = UNet(out_classes=1)
    
    # %%
    
    import torch
    from segmentation_models_pytorch.losses import JaccardLoss
    
    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
    TRAINING = True

    # Set num of epochs
    EPOCHS = 12

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function
    loss_fn = JaccardLoss('binary')

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.00008),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # %%
    
    device = torch.device("cpu")
    
    model.train()
    
    for epoch in range(0, 2):
        for batch_idx, this_batch in enumerate(train_dataloader):
            data = this_batch['image']
            target = this_batch['labels']
            
            # data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # %%
    
    ##### BELOW: Experiment with pre-computed Unet
    
    
    # %%
    
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )

    for param in model.parameters():
        param.requires_grad = True
        
    model.train()
        
    # %%
    from torchmetrics import JaccardIndex
    
    criterion = JaccardIndex(task="binary")
    
    # %%
    
    import torch.optim as optim
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # %%
    
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image']
            labels = data['labels']
            labels = labels.reshape(1, 1, 384, 384)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
# %%
