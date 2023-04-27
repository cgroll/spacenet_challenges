# %%
from src.path import ProjPaths
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import rioxarray
from PIL import Image
import numpy as np
from numpy import asarray

from src.file_tools import get_all_files_in_path, extract_img_id_from_str, derive_geo_raster_file_names_from_image_ids

# %%

class Band3BinaryMaskDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir # train, test, ...
        
        img_file_list = get_all_files_in_path(self.root_dir / "3band")
        
        self.image_ids = [extract_img_id_from_str(fname) for fname in img_file_list]
        self.files_3band = img_file_list
        self.files_geo_raster = derive_geo_raster_file_names_from_image_ids(self.image_ids)
        
        self.transform = transform
    
    def __len__(self):
        
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file_path = self.root_dir / "3band" / self.files_3band[idx]
        geo_raster_file_path = self.root_dir / "geo_raster_3band" / self.files_geo_raster[idx]
        
        img_darr = rioxarray.open_rasterio(img_file_path)
        img_data = img_darr.values / 256
        img_data = np.transpose(img_data, (1, 2, 0))
        
        rasterized_reloaded = Image.open(geo_raster_file_path)
        rasterized_reloaded_np = asarray(rasterized_reloaded)
        
        sample = {'image': img_data, 'labels': rasterized_reloaded_np * 1.0}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class RandomCropImgAndLabels(object):
    """
    Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        if (h < self.output_size[0]) | (w < self.output_size[1]): # TODO: check size[0], size[1]
            raise ValueError(f'Image already smaller than requested crop size: width {w}, height {h}')

        if h == new_h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        labels = labels[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'labels': labels}

class UpperLeftCropImgAndLabels(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        if (h < self.output_size[0]) | (w < self.output_size[1]): # TODO: check size[0], size[1]
            raise ValueError(f'Image already smaller than requested crop size: width {w}, height {h}')

        top = 0
        left = 0

        image = image[top: top + new_h,
                      left: left + new_w]

        labels = labels[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'labels': labels}

class ToTensorImgAndLabels(object):
    """
    Convert ndarrays in sample to Tensors.
    Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __call__(self, sample, extend=False):
        image, labels = sample['image'], sample['labels']

        labels_torch = torch.from_numpy(labels.astype('float32'))
        labels_torch = labels_torch[None, :]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image.astype('float32')),
                'labels': labels_torch} # TODO: change back to "labels" or use consistently

class ToTensorImgAndLabelsExtension(object):
    """
    Convert ndarrays in sample to Tensors.
    Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __call__(self, sample, extend=False):
        image, labels = sample['image'], sample['labels']

        labels_torch = torch.from_numpy(labels.astype('float32'))
        labels_torch = labels_torch[None, :]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image.astype('float32')),
                'labels': labels_torch} # TODO: change back to "labels" or use consistently


if __name__ == '__main__':
    
    train_path = ProjPaths.interim_sn1_data_path / "train"
    
    train_dataset = Band3BinaryMaskDataset(train_path, transform=transforms.Compose([
                                               RandomCropImgAndLabels(406),
                                               ToTensorImgAndLabels()
                                           ]))
    
    # this_sample = train_dataset[5]
    
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['labels'].size())

        # observe 4th batch and stop.
        if i_batch == 300000:
            break
            # plt.figure()
            # show_landmarks_batch(sample_batched)
            # plt.axis('off')
            # plt.ioff()
            # plt.show()
