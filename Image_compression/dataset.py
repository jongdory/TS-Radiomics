import random
import os
import pandas as pd
import numpy as np
import math
import torch
import SimpleITK as sitk
from sklearn.model_selection import KFold
from monai import transforms
from torch.utils.data import Dataset
from scipy.ndimage import center_of_mass, distance_transform_edt
from monai.data import Dataset as MonaiDataset


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


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


def get_brats_dataset(data_path):
    transform = brats_transforms

    data = []
    for i, subject in enumerate(os.listdir(data_path)):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz")
        seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")
        data.append({"image":t1ce, "seg":seg})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

class BrainDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.imgs = get_brats_dataset(data_path)

    def __len__(self):
        return len(self.imgs)
    

    def __getitem__(self, i):
        x = self.imgs[i]['image']
        x_seg = self.imgs[i]['seg']
        
        subject_id = self.imgs[i][f"image_meta_dict"]['filename_or_obj'].split('/')[-1].split('_')[1]
        return {"x": x , "x_seg": x_seg, "subject_id": subject_id}