import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from src.data.band3_binary_mask_data import Band3BinaryMaskDataset, RandomCropImgAndLabels, ToTensorImgAndLabels, ToTensorImgAndLabelsExtension
from src.path import ProjPaths


IMAGE_PIXEL_SIZE = 384

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


# from https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch

class DoubleConv(nn.Module):
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
    
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

    
class UpBlock(nn.Module):
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
    def __init__(self, out_classes=1, up_sample_mode='conv_transpose', lr=0.00008, optimizer='Adam'):
        super(UNet, self).__init__()

        # self.save_hyperparameters()
        
        # implementation of UNet with 1 layers less than original UNet. Usually there should be 5 down and up blocks.
        # see: https://www.kaggle.com/code/alexj21/pytorch-eda-unet-from-scratch-finetuning

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
        self.optimizer = optimizer

        self.example_input_array = torch.rand(1, 3, IMAGE_PIXEL_SIZE, IMAGE_PIXEL_SIZE)

    # def on_train_start(self):
    #     self.logger.log_hyperparams(self.hparams, {'max_epochs': self.trainer.max_epochs, 'hp_metric': -3})

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
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam([dict(params=self.parameters(), lr=self.lr),])
        else:
            raise ValueError(f'Optimizer {self.optimizer} is not implemented yet')
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch['image']
        target = train_batch['labels']

        logits = self.forward(data)
        loss = F.binary_cross_entropy_with_logits(
            input=logits, target=target
        )
        
        #self.log('train_loss', loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch['image']
        target = val_batch['labels']

        logits = self.forward(data)
        loss = F.binary_cross_entropy_with_logits(
            input=logits, target=target
        )

        # compute metrics
        threshold = 0.5
        normed_logits = torch.sigmoid(logits).cpu().numpy()
        true_labels = target.cpu().numpy()

        normed_logits[normed_logits >= threshold] = 1
        normed_logits[normed_logits < threshold] = 0

        n_correct = (normed_logits == true_labels).sum()
        n_entries = np.prod(normed_logits.shape)
        accuracy = n_correct / n_entries

        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        self.log('hp_metric', loss)
        # self.logger.log_metrics({'val_loss': loss})



if __name__ == '__main__':
    
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    
    MAX_EPOCHS = 20 # 20
    OPTIMIZER = 'Adam'
    LOSS_FUNC = ''

    data_module = UNetDataModule()
    
    from pathlib import PureWindowsPath

    for this_lr in [0.0002]: #[0.00008, 0.0001]: # [0.00008, 0.0001, 0.00015]: #, 0.0002

        logger = TensorBoardLogger('tb_logs_2', name='UNet Ptl V1', log_graph=True)
        logger.log_hyperparams({"max_epochs": MAX_EPOCHS, "optimizer": OPTIMIZER, 'lr': this_lr})

        # model = UNet(lr=this_lr, optimizer=OPTIMIZER)

        chkpt_path = str(PureWindowsPath("C:\\Users\\cgrol\\OneDrive\\Dokumente\\GitHub\\spacenet_challenges\\tb_logs_2\\UNet Ptl V1\\version_3\\checkpoints\\epoch=15-step=13376.ckpt"))
        model = UNet.load_from_checkpoint(chkpt_path)
        #logger.log_graph(model, input_array=)
        # trainer = pl.Trainer(max_epochs=2, precision=16, accelerator="gpu", logger=logger)
        # trainer = pl.Trainer(precision=16, accelerator="gpu", logger=logger)
        # trainer = pl.Trainer(max_epochs=MAX_EPOCHS, precision=16, accelerator="gpu", 
        #                      logger=logger, limit_train_batches=10, log_every_n_steps=2, 
        #                      callbacks=[checkpoint_callback_best, checkpoint_callback_default])

        checkpoint_callback_best = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            #dirpath='model_checkpoints/unet',
            filename='best_model-unet-{epoch:02d}-{val_loss:.2f}') # -lr_{lr:2.8f}

        checkpoint_callback_default = ModelCheckpoint()

        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, precision='16-mixed', accelerator="gpu", 
                             logger=logger,
                             callbacks=[checkpoint_callback_best, checkpoint_callback_default, 
                                        EarlyStopping(monitor="val_loss", mode="min", patience=5)
                                        ])
        # trainer = pl.Trainer(max_epochs=2, precision=16, accelerator="gpu", fast_dev_run=True)

        trainer.fit(model, data_module)
        # trainer.fit(model, ckpt_path=chkpt_path)

        logger.finalize("success")
        print(checkpoint_callback_best.best_model_path)



    # TODO:
    # - log images
    # - log into different subfolders (ADAM vs SDG)
    # - resume training -> There was a problem with data_loader; but loading model weights works
    # - fix hp_metric -> Done
    # - try finding optimal learning rate
    # - try mixed precision -> didn't reduce memory size for me. Hence, could not further increase batch size
    # - try different batch sizes
    
    # Other models, data usage:
    # - fine-tuned model
    # - include multi-spectral data
    # - include other data like open maps?