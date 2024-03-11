
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
    def __init__(self, dataset_path, radiomics_path, img_size=64, resize = None):
        super().__init__()
        self.imgs = get_brats_dataset(dataset_path, radiomics_path)
        self.size = img_size
        self.resize = resize
        if resize is not None:
            self.transform = get_brats_transform(resize)

    def __len__(self):
        return len(self.imgs)
    
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

    def __getitem__(self, i):
        s = self.size//2
        mask = self.imgs[i]['mask']
        mask_clone = mask.clone()
        mask_clone[mask_clone == 1] = 1
        mask_clone[mask_clone == 4] = 1
        mask_clone[mask_clone != 1] = 0
        cx, cy, cz = center_of_mass(mask_clone[0])
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - s, cy - s, cz - s
        ex, ey, ez = cx + s, cy + s, cz + s
        if sx < 0: sx, ex = 0, s*2
        if sy < 0: sy, ey = 0, s*2
        if sz < 0: sz, ez = 0, s*2
        if ex > 192: sx, ex = 192-s*2, 192
        if ey > 192: sy, ey = 192-s*2, 192
        if ez > 160: sz, ez = 160-s*2, 160
        crop_mask = mask[0,sx:ex, sy:ey, sz:ez]

        crop_mask = torch.stack([crop_mask == 1, crop_mask == 4], dim = 0).float()
        crop_mask[0] = crop_mask[0]
        crop_mask[1] = crop_mask[0] + crop_mask[1]

        if self.resize is not None:
            crop_mask = self.transform({"mask":crop_mask.numpy()})["mask"]
        crop_mask = brats_aug_transforms({"mask":crop_mask.numpy()})["mask"]
        crop_mask = crop_mask.float()
        
        return {"x": crop_mask[None], "radiomics": self.imgs[i]['radiomics']}