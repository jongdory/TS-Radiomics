
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms as T

from beartype.door import is_bearable
from beartype.typing import Tuple

from monai import transforms
from monai.data import Dataset as MonaiDataset
from scipy.ndimage import center_of_mass
from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def convert_image_to_fn(img_type, image):
    if image.mode == img_type:
        return image

    return image.convert(img_type)

# custom collation function
# so dataset can return a str and it will collate into List[str]

def collate_tensors_or_str(data):
    is_one_data = not isinstance(data[0], tuple)

    if is_one_data:
        # data = torch.stack(data)
        return data[0]['x'], data[0]['radiomics']

    outputs = []
    for datum in zip(*data):
        if is_bearable(datum, Tuple[str, ...]):
            output = list(datum)
        else:
            output = torch.stack(datum)

        outputs.append(output)

    return tuple(outputs)

# dataset classes

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        assert len(self.paths) > 0, 'your folder contains no images'
        assert len(self.paths) > 100, 'you need at least 100 images, 10k for research paper, millions for miraculous results (try Laion-5B)'

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, shuffle = True, drop_last = True, **kwargs)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class TextImageDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

class MockTextImageDataset(TextImageDataset):
    def __init__(
        self,
        image_size,
        length = int(1e5),
        channels = 3
    ):
        self.image_size = image_size
        self.channels = channels
        self.length = length

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        mock_image = torch.randn(self.channels, self.image_size, self.image_size)
        return mock_image, 'mock text'

# --- brats dataset ---
brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys="mask"),
        transforms.EnsureChannelFirstd(keys="mask"),
        transforms.Lambdad(keys="mask", func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys="mask"),
        transforms.EnsureTyped(keys="mask"),
        transforms.Orientationd(keys="mask", axcodes="RAI"),
        transforms.CropForegroundd(keys="mask", source_key="mask"),
        transforms.CenterSpatialCropd(keys="mask", roi_size=(192, 192, 160)),
        transforms.SpatialPadd(keys="mask", spatial_size=(192, 192, 160), allow_missing_keys=True),
        # transforms.ScaleIntensityRangePercentilesd(keys="mask", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)

brats_aug_transforms = transforms.Compose(
    [
        transforms.RandRotate90d(keys="mask", prob=0.2, spatial_axes=(0,1)),
        transforms.RandRotate90d(keys="mask", prob=0.2, spatial_axes=(0,2)),
        transforms.RandRotate90d(keys="mask", prob=0.2, spatial_axes=(1,2)),
        transforms.RandRotated(keys="mask", prob=0.4, range_x=0.3, range_y=0.3, range_z=0.3, mode='nearest'),
        transforms.RandAxisFlipd(keys="mask", prob=0.3),
    ]
)

def get_brats_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resized(keys="mask", spatial_size=(img_size[0], img_size[1], img_size[2]), mode='nearest'),
        ]
    )

def get_brats_dataset(data_path, radiomics_path):
    transform = brats_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == subject].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        mask = os.path.join(sub_path, f"{subject}_seg.nii.gz")

        data.append({"mask":mask, 'radiomics': radiomics})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)


class BraTSDataset(Dataset):
    def __init__(self, dataset_path, radiomics_path, resize = None):
        super().__init__()
        self.imgs = get_brats_dataset(dataset_path, radiomics_path)
        self.resize = resize
        if resize is not None:
            self.transform = get_brats_transform(resize)

    def __len__(self):
        return len(self.imgs)
    
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

    def __getitem__(self, i):
        mask = self.imgs[i]['mask']
        mask_clone = mask.clone()
        mask_clone[mask_clone == 1] = 1
        mask_clone[mask_clone == 4] = 1
        mask_clone[mask_clone != 1] = 0
        cx, cy, cz = center_of_mass(mask_clone[0])
        cx, cy, cz = int(cx), int(cy), int(cz)
        # sx, sy, sz = cx - 48, cy - 48, cz - 48
        # ex, ey, ez = cx + 48, cy + 48, cz + 48
        # if sx < 0: sx, ex = 0, 96
        # if sy < 0: sy, ey = 0, 96
        # if sz < 0: sz, ez = 0, 96
        # if ex > 192: sx, ex = 96, 192
        # if ey > 192: sy, ey = 96, 192
        # if ez > 160: sz, ez = 64, 160
        sx, sy, sz = cx - 32, cy - 32, cz - 32
        ex, ey, ez = cx + 32, cy + 32, cz + 32
        if sx < 0: sx, ex = 0, 64
        if sy < 0: sy, ey = 0, 64
        if sz < 0: sz, ez = 0, 64
        if ex > 192: sx, ex = 128, 192
        if ey > 192: sy, ey = 128, 192
        if ez > 160: sz, ez = 96, 160
        # mask = mask[:,sx:ex, sy:ey, sz:ez]
        crop_mask = mask[0,sx:ex, sy:ey, sz:ez]

        # # crop mask to 3 channel brats mask mapping
        # crop_mask = torch.stack([crop_mask == 1, crop_mask == 4, crop_mask == 2], dim = 0).float()
        # crop_mask[0] = crop_mask[0]
        # crop_mask[1] = crop_mask[0] + crop_mask[1]
        # crop_mask[2] = crop_mask[1] + crop_mask[2]
        # crop mask to 2 channel brats mask mapping
        crop_mask = torch.stack([crop_mask == 1, crop_mask == 4], dim = 0).float()
        crop_mask[0] = crop_mask[0]
        crop_mask[1] = crop_mask[0] + crop_mask[1]

        if self.resize is not None:
            crop_mask = self.transform({"mask":crop_mask.numpy()})["mask"]
        crop_mask = brats_aug_transforms({"mask":crop_mask.numpy()})["mask"]
        crop_mask = crop_mask.float()
        
        return {"x": crop_mask[None], "radiomics": self.imgs[i]['radiomics']}
    
class BraTSDataset2(Dataset):
    def __init__(self, dataset_path, radiomics_path, resize = None):
        super().__init__()
        self.imgs = get_brats_dataset(dataset_path, radiomics_path)
        self.resize = resize
        if resize is not None:
            self.transform = get_brats_transform(resize)

    def __len__(self):
        return len(self.imgs)
    
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

    def __getitem__(self, i):
        mask = self.imgs[i]['mask']
        mask_clone = mask.clone()
        mask_clone[mask_clone == 1] = 1
        mask_clone[mask_clone == 4] = 1
        mask_clone[mask_clone != 1] = 0
        cx, cy, cz = center_of_mass(mask_clone[0])
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 48, cy - 48, cz - 48
        ex, ey, ez = cx + 48, cy + 48, cz + 48
        if sx < 0: sx, ex = 0, 96
        if sy < 0: sy, ey = 0, 96
        if sz < 0: sz, ez = 0, 96
        if ex > 192: sx, ex = 96, 192
        if ey > 192: sy, ey = 96, 192
        if ez > 160: sz, ez = 64, 160
        # mask = mask[:,sx:ex, sy:ey, sz:ez]
        crop_mask = mask[0,sx:ex, sy:ey, sz:ez]

        # crop mask to 2 channel brats mask mapping
        crop_mask = torch.stack([crop_mask == 1, crop_mask == 4], dim = 0).float()
        crop_mask[0] = crop_mask[0]
        crop_mask[1] = crop_mask[0] + crop_mask[1]

        if self.resize is not None:
            crop_mask = self.transform({"mask":crop_mask.numpy()})["mask"]
        crop_mask = brats_aug_transforms({"mask":crop_mask.numpy()})["mask"]
        crop_mask = crop_mask.float()
        
        return {"x": crop_mask[None], "radiomics": self.imgs[i]['radiomics']}
    
# --- lung dataset ---
lung_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys="mask"),
        transforms.EnsureChannelFirstd(keys="mask"),
        transforms.Lambdad(keys="mask", func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys="mask"),
        transforms.EnsureTyped(keys="mask"),
        transforms.Orientationd(keys="mask", axcodes="RAI"),
        transforms.CropForegroundd(keys="mask", source_key="mask"),
        transforms.CenterSpatialCropd(keys="mask", roi_size=(224, 224, 96)),
        transforms.SpatialPadd(keys="mask", spatial_size=(224, 224, 96), allow_missing_keys=True),
        # transforms.ScaleIntensityRangePercentilesd(keys="mask", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)

def get_lung_dataset(data_path, radiomics_path):
    transform = lung_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == subject].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        mask = os.path.join(sub_path, "crop_tumor.nii.gz")

        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask))
        if np.sum(mask_arr) == 0: continue

        data.append({"mask":mask, 'radiomics': radiomics})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)
    
class LungDataset(Dataset):
    def __init__(self, dataset_path, radiomics_path, resize=None):
        super().__init__()
        self.imgs = get_lung_dataset(dataset_path, radiomics_path)
        self.resize = resize
        if resize is not None:
            self.transform = get_brats_transform(resize)

    def __len__(self):
        return len(self.imgs)
    
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

    def __getitem__(self, i):
        mask = self.imgs[i]['mask']
        cx, cy, cz = center_of_mass(mask[0])
        cx, cy, cz = int(cx), int(cy), int(cz)
        # sx, sy, sz = cx - 64, cy - 64, cz - 48
        # ex, ey, ez = cx + 64, cy + 64, cz + 48
        # if sx < 0: sx, ex = 0, 128
        # if sy < 0: sy, ey = 0, 128
        # if sz < 0: sz, ez = 0, 96
        # if ex > 224: sx, ex = 96, 224
        # if ey > 224: sy, ey = 96, 224
        # if ez > 32: sz, ez = 0, 96
        sx, sy, sz = cx - 48, cy - 48, cz - 48
        ex, ey, ez = cx + 48, cy + 48, cz + 48
        if sx < 0: sx, ex = 0, 96
        if sy < 0: sy, ey = 0, 96
        if sz < 0: sz, ez = 0, 96
        if ex > 224: sx, ex = 128, 224
        if ey > 224: sy, ey = 128, 224
        if ez > 32: sz, ez = 0, 96
        crop_mask = mask[:, sx:ex, sy:ey, sz:ez]

        if self.resize is not None:
            crop_mask = self.transform({"mask":crop_mask.numpy()})["mask"]
        crop_mask = brats_aug_transforms({"mask":crop_mask.numpy()})["mask"]

        return {"x": crop_mask[None], "radiomics": self.imgs[i]['radiomics']}
    
lung_aug_transforms = transforms.Compose(
    [
        transforms.RandRotate90d(keys="mask", prob=0.4, spatial_axes=(0,1)),
    ]
)

class LungDataset2(Dataset):
    def __init__(self, dataset_path, radiomics_path, resize=None):
        super().__init__()
        self.imgs = get_lung_dataset(dataset_path, radiomics_path)
        self.resize = resize
        if resize is not None:
            self.transform = get_brats_transform(resize)

    def __len__(self):
        return len(self.imgs)
    
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

    def __getitem__(self, i):
        mask = self.imgs[i]['mask']
        cx, cy, cz = center_of_mass(mask[0])
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 64, cy - 64, cz - 48
        ex, ey, ez = cx + 64, cy + 64, cz + 48
        if sx < 0: sx, ex = 0, 128
        if sy < 0: sy, ey = 0, 128
        if sz < 0: sz, ez = 0, 96
        if ex > 224: sx, ex = 96, 224
        if ey > 224: sy, ey = 96, 224
        if ez > 32: sz, ez = 0, 96
        crop_mask = mask[:, sx:ex, sy:ey, sz:ez]

        if self.resize is not None:
            crop_mask = self.transform({"mask":crop_mask.numpy()})["mask"]
        crop_mask = lung_aug_transforms({"mask":crop_mask.numpy()})["mask"]

        return {"x": crop_mask[None], "radiomics": self.imgs[i]['radiomics']}
    
# breast dataset
def get_breast_dataset(data_path, radiomics_path):
    transform = brats_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == int(subject)].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        mask = os.path.join(sub_path, "roi.nii.gz")

        data.append({"mask":mask, 'radiomics': radiomics})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

class BreastDataset(Dataset):
    def __init__(self, dataset_path, radiomics_path, resize=None):
        super().__init__()
        self.imgs = get_breast_dataset(dataset_path, radiomics_path)
        self.resize = resize
        if resize is not None:
            self.transform = get_brats_transform(resize)

    def __len__(self):
        return len(self.imgs)
    
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)
    
    def get_mask(self, mask):
        mask_clone = mask.clone()
        mask_clone[mask_clone == 1] = 1
        mask_clone[mask_clone == 2] = 1
        mask_clone[mask_clone != 1] = 0
        cx, cy, cz = center_of_mass(mask_clone[0])
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 32, cy - 32, cz - 32
        ex, ey, ez = cx + 32, cy + 32, cz + 32
        if sx < 0: sx, ex = 0, 64
        if sy < 0: sy, ey = 0, 64
        if sz < 0: sz, ez = 0, 64
        if ex > 192: sx, ex = 128, 192
        if ey > 192: sy, ey = 128, 192
        if ez > 160: sz, ez = 96, 160
        crop_mask = mask[0, sx:ex, sy:ey, sz:ez]
        crop_mask = torch.stack([crop_mask == 2, crop_mask == 1], dim = 0).float()
        crop_mask[0] = crop_mask[0]
        crop_mask[1] = crop_mask[0] + crop_mask[1]

        return crop_mask


    def __getitem__(self, i):
        mask = self.imgs[i]['mask']
        mask = self.get_mask(mask)

        if self.resize is not None:
            mask = self.transform({"mask":mask.numpy()})["mask"]
        mask = brats_aug_transforms({"mask":mask.numpy()})["mask"]

        return {"x": mask[None], "radiomics": self.imgs[i]['radiomics']}
    

class BreastDataset2(Dataset):
    def __init__(self, dataset_path, radiomics_path, resize=None):
        super().__init__()
        self.imgs = get_breast_dataset(dataset_path, radiomics_path)
        self.resize = resize
        if resize is not None:
            self.transform = get_brats_transform(resize)

    def __len__(self):
        return len(self.imgs)
    
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)
    
    def get_mask(self, mask):
        mask_clone = mask.clone()
        mask_clone[mask_clone == 1] = 1
        mask_clone[mask_clone == 2] = 1
        mask_clone[mask_clone != 1] = 0
        cx, cy, cz = center_of_mass(mask_clone[0])
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 48, cy - 48, cz - 48
        ex, ey, ez = cx + 48, cy + 48, cz + 48
        if sx < 0: sx, ex = 0, 96
        if sy < 0: sy, ey = 0, 96
        if sz < 0: sz, ez = 0, 96
        if ex > 192: sx, ex = 96, 192
        if ey > 192: sy, ey = 96, 192
        if ez > 160: sz, ez = 64, 160
        crop_mask = mask[0, sx:ex, sy:ey, sz:ez]
        crop_mask = torch.stack([crop_mask == 2, crop_mask == 1], dim = 0).float()
        crop_mask[0] = crop_mask[0]
        crop_mask[1] = crop_mask[0] + crop_mask[1]

        return crop_mask


    def __getitem__(self, i):
        mask = self.imgs[i]['mask']
        mask = self.get_mask(mask)

        if self.resize is not None:
            mask = self.transform({"mask":mask.numpy()})["mask"]
        mask = brats_aug_transforms({"mask":mask.numpy()})["mask"]

        return {"x": mask[None], "radiomics": self.imgs[i]['radiomics']}
    
#kidney dataseet
kit23_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys="mask"),
        transforms.EnsureChannelFirstd(keys="mask"),
        transforms.Lambdad(keys="mask", func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys="mask"),
        transforms.EnsureTyped(keys="mask"),
        transforms.Orientationd(keys="mask", axcodes="RAI"),
        transforms.CropForegroundd(keys="mask", select_fn=lambda x: x == 2, source_key="mask"),
        transforms.CenterSpatialCropd(keys="mask", roi_size=(112, 112, 96)),
        transforms.SpatialPadd(keys="mask", spatial_size=(112, 112, 96), allow_missing_keys=True),
        # transforms.ScaleIntensityRangePercentilesd(keys="mask", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)

def get_kidney_dataset(data_path, radiomics_path):
    transform = kit23_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    phase = ['left', 'right']
    for i, subject in enumerate(os.listdir(data_path)):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        for p in phase:
            try: radiomics =  radiomics_df[(radiomics_df["subject"] == subject) & (radiomics_df["phase"] == p)].values[0][2:].astype(np.float32)
            except: continue
            mask = os.path.join(sub_path, f"{p}_seg.nii.gz")
            if not os.path.exists(mask): continue

            data.append({"mask":mask, 'radiomics': radiomics})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)


class KidneyDataset(Dataset):
    def __init__(self, dataset_path, radiomics_path, resize=None):
        super().__init__()
        self.imgs = get_kidney_dataset(dataset_path, radiomics_path)
        self.resize = resize
        if resize is not None:
            self.transform = get_brats_transform(resize)

    def __len__(self):
        return len(self.imgs)
    
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)
    
    def get_mask(self, mask):
        mask_clone = mask.clone()
        mask_clone[mask_clone != 2] = 0
        mask_clone[mask_clone == 2] = 1
        mx, my, mz = mask.shape[1], mask.shape[2], mask.shape[3]
        if mz < 96:
            mask_clone = torch.zeros((1, mx, my, 96))
            cen = (96 - mz) // 2
            mask_clone[:, :, :, cen:cen+mz] = mask
            mask = mask_clone

        cx, cy, cz = center_of_mass(mask_clone[0])
        # if np.isnan(cx): cx = mx//2
        # if np.isnan(cy): cy = my//2
        # if np.isnan(cz): cz = mz//2
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 48, cy - 48, cz - 48
        ex, ey, ez = cx + 48, cy + 48, cz + 48
        if sx < 0: sx, ex = 0, 96
        if sy < 0: sy, ey = 0, 96
        if sz < 0: sz, ez = 0, 96
        if ex > mx: sx, ex = mx-96, mx
        if ey > my: sy, ey = my-96, my
        if ez > mz: sz, ez = mz-96, mz
        crop_mask = mask[:, sx:ex, sy:ey, sz:ez]

        crop_mask[crop_mask != 2] = 0
        crop_mask[crop_mask == 2] = 1

        return crop_mask


    def __getitem__(self, i):
        mask = self.imgs[i]['mask']
        mask = self.get_mask(mask)

        if self.resize is not None:
            mask = self.transform({"mask":mask.numpy()})["mask"]
        mask = lung_aug_transforms({"mask":mask.numpy()})["mask"]

        return {"x": mask[None], "radiomics": self.imgs[i]['radiomics']}
    
