import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai import transforms
from monai.data import Dataset as MonaiDataset

from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
from scipy.ndimage import center_of_mass, distance_transform_edt


brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "seg"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "seg"], allow_missing_keys=True),
        transforms.Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image", "seg"], axcodes="RAI", allow_missing_keys=True),
        transforms.CenterSpatialCropd(keys=["image", "seg"], roi_size=(192, 192, 160), allow_missing_keys=True),
        transforms.SpatialPadd(keys=["image", "seg"], spatial_size=(192, 192, 160), allow_missing_keys=True),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)

def get_brats_dataset(data_path, radiomics_path):
    transform = brats_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    for i, subject in enumerate(os.listdir(data_path)):
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == subject].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz")
        seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")
        data.append({"image":t1ce, "seg":seg, 'radiomics': radiomics})
        data.append({"image":t1ce, "seg":seg, 'radiomics': np.array([0], dtype=np.float32)})

    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

@Registers.datasets.register_with_name('brats')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_brats_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask == 1] = 1
        mask[mask != 1] = 0
        cx, cy, cz = center_of_mass(mask[0].numpy())
        mx, my, mz = mask.shape[1], mask.shape[2], mask.shape[3]
        if np.isnan(cx): cx = 96
        if np.isnan(cy): cy = 96
        if np.isnan(cz): cz = 80
        cx, cy, cz = int(cx), int(cy), int(cz)

        sx, sy, sz = cx - 56, cy - 56, cz - 48
        ex, ey, ez = cx + 56, cy + 56, cz + 48
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 96
        if ex > mx: sx, ex = mx - 112, mx
        if ey > my: sy, ey = my - 112, my
        if ez > mz: sz, ez = mz - 96, mz
        
        return sx, sy, sz, ex, ey, ez
    
    def random_rotate_flip(self, x, x_cond):
        axis = [[1, 2], [1, 3], [2, 3]]
        if random.random() > 0.5:
            rotate = random.randint(1, 3)
            ax = random.randint(0, 2)
            x = torch.rot90(x, rotate, axis[ax])
            x_cond = torch.rot90(x_cond, rotate, axis[ax])
        if random.random() > 0.5:
            flip = random.randint(1, 3)
            x = torch.flip(x, [flip])
            x_cond = torch.flip(x_cond, [flip])

        return x, x_cond

    def tumor_crop(self, x, x_cond, x_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_cond[x_seg == 1] = 0.5
        x_cond[x_seg == 4] = 1

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        x_seg[x_seg == 2] = 0
        x_seg[x_seg == 1] = 0.5
        x_seg[x_seg == 4] = 1

        return x, x_cond, x_seg
    
    def non_tumor_crop(self, x, x_cond, x_seg):
        tumor_mask = (x_seg[0] == 1) | (x_seg[0] == 4)
        brain_mask = (x[0] > 0)
        
        distance_map = distance_transform_edt(np.logical_and(brain_mask, ~tumor_mask))
        cz, cy, cx = np.unravel_index(distance_map.argmax(), distance_map.shape)
        
        sx, sy, sz = cx - 56, cy - 56, cz - 48
        ex, ey, ez = cx + 56, cy + 56, cz + 48
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 96
        if ex > 192: sx, ex = 80, 192
        if ey > 192: sy, ey = 80, 192
        if ez > 160: sz, ez = 64, 160

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

        x_cond[x_seg == 1] = 0.5
        x_cond[x_seg == 4] = 1
        
        return x, x_cond

    def __getitem__(self, i):
        x = self.imgs[i]['image']
        x_cond = self.imgs[i]['image']
        x_seg = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']
        
        if x_rad.shape[0] == 1:
            x, x_cond = self.non_tumor_crop(x, x_cond, x_seg)
            x, x_cond = self.random_rotate_flip(x, x_cond)
        else:
            x, x_cond, x_seg = self.tumor_crop(x, x_cond, x_seg)

        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-1].split('_')[1]
        return {"x": x , "x_cond": x_cond, "x_seg":x_seg, "radiomics": x_rad, "subject_id": subject_id}