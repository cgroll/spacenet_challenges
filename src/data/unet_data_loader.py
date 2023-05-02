from src.data.band3_binary_mask_data import Band3BinaryMaskDataset, RandomCropImgAndLabels, ToTensorImgAndLabels, SamDistTransfDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.path import ProjPaths
from src.config_vars import ConfigVariables

class UNetDataModule(pl.LightningDataModule):

    def setup(self, stage):
        train_path = ProjPaths.interim_sn1_data_path / "train"
        val_path = ProjPaths.interim_sn1_data_path / "val"

        IMAGE_PIXEL_SIZE = ConfigVariables.unet_3band_img_pixel_size
    
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

class UNetDistTransfSamDataModule(pl.LightningDataModule):

    def setup(self, stage):
        train_path = ProjPaths.interim_sn1_data_path / "train"
        val_path = ProjPaths.interim_sn1_data_path / "val"

        IMAGE_PIXEL_SIZE = ConfigVariables.unet_3band_img_pixel_size
    
        self.train_dataset = SamDistTransfDataset(train_path, transform=transforms.Compose([
                                                RandomCropImgAndLabels(IMAGE_PIXEL_SIZE),
                                                ToTensorImgAndLabels()
                                            ]))
        self.val_dataset = SamDistTransfDataset(val_path, transform=transforms.Compose([
                                                RandomCropImgAndLabels(IMAGE_PIXEL_SIZE),
                                                ToTensorImgAndLabels()
                                            ]))
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=6, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=6, shuffle=False, num_workers=1)
