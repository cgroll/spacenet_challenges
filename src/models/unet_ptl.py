import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data.band3_binary_mask_data import Band3BinaryMaskDataset, RandomCropImgAndLabels, ToTensorImgAndLabels, ToTensorImgAndLabelsExtension
from src.path import ProjPaths


IMAGE_PIXEL_SIZE = 384

# from https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch

class DoubleConv(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class DownBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

    
class UpBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

    
class UNet(pl.LightningModule):
    def __init__(self, out_classes=1, up_sample_mode='conv_transpose', lr=0.00008):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

        self.lr = lr

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x
    
    def BCELoss(self, logits, labels):
        return nn.BCEWithLogitsLoss(logits, labels)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([dict(params=self.parameters(), lr=self.lr),])
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch['image']
        target = train_batch['labels']

        logits = self.forward(data)
        loss = F.binary_cross_entropy_with_logits(
            input=logits, target=target
        )
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch['image']
        target = val_batch['labels']

        logits = self.forward(data)
        loss = F.binary_cross_entropy_with_logits(
            input=logits, target=target
        )

        self.log('val_loss', loss)


class UNetDataModule(pl.LightningDataModule):

    def setup(self, stage):
        train_path = ProjPaths.interim_sn1_data_path / "train"
        val_path = ProjPaths.interim_sn1_data_path / "val"
    
        self.train_dataset = Band3BinaryMaskDataset(train_path, transform=transforms.Compose([
                                                RandomCropImgAndLabels(IMAGE_PIXEL_SIZE),
                                                ToTensorImgAndLabels()
                                            ]))
        self.val_dataset = Band3BinaryMaskDataset(val_path, transform=transforms.Compose([
                                                RandomCropImgAndLabels(IMAGE_PIXEL_SIZE),
                                                ToTensorImgAndLabels()
                                            ]))
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=6, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=6, shuffle=False, num_workers=1)


if __name__ == '__main__':
    
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger('tb_logs', name='UNet Ptl V1')

    data_module = UNetDataModule()

    # train

    for lr in [0.00008, 0.0001, 0.00015, 0.0002]:
        model = UNet(lr=lr)
        # trainer = pl.Trainer(max_epochs=2, precision=16, accelerator="gpu", logger=logger)
        trainer = pl.Trainer(precision=16, accelerator="gpu", logger=logger)
        # trainer = pl.Trainer(max_epochs=2, precision=16, accelerator="gpu", logger=logger, limit_train_batches=10, log_every_n_steps=2)
        # trainer = pl.Trainer(max_epochs=2, precision=16, accelerator="gpu", fast_dev_run=True)

        trainer.fit(model, data_module)
