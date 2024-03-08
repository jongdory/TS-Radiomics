import random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
from monai import transforms
from monai.data import Dataset as MonaiDataset

from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
from PIL import Image
from scipy.ndimage import center_of_mass, distance_transform_edt
import cv2
import os
import SimpleITK as sitk
from scipy.ndimage import zoom
from einops import rearrange


def find_brain_bounds(x):
    # Assuming brain tissues have values greater than 0 (as background is 0)
    brain_mask = x > 0
    bx, ex = np.where(brain_mask.sum(axis=1).sum(axis=1) != 0)[0][[0, -1]]
    by, ey = np.where(brain_mask.sum(axis=0).sum(axis=1) != 0)[0][[0, -1]]
    bz, ez = np.where(brain_mask.sum(axis=0).sum(axis=0) != 0)[0][[0, -1]]
    return bx, ex, by, ey, bz, ez

@Registers.datasets.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]


@Registers.datasets.register_with_name('custom_colorization_LAB')
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1., 1.)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name('custom_colorization_RGB')
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        cond_image = image.convert('L')
        cond_image = cond_image.convert('RGB')

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            cond_image = (cond_image - 0.5) * 2.
            cond_image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name('custom_inpainting')
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.
        if index >= self._length:
            index = index - self._length
            p = 1.

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[:, mask_pos_x:mask_pos_x+mask_height, mask_pos_y:mask_pos_y+mask_width] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


# ---3D brats dataset ---

brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "seg", "next_seg"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "seg", "next_seg"], allow_missing_keys=True),
        transforms.Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image", "seg", "next_seg"], axcodes="RAI", allow_missing_keys=True),
        # transforms.CropForegroundd(keys=["image", "seg"], source_key="image", allow_missing_keys=True),
        transforms.CenterSpatialCropd(keys=["image", "seg", "next_seg"], roi_size=(192, 192, 160), allow_missing_keys=True),
        transforms.SpatialPadd(keys=["image", "seg", "next_seg"], spatial_size=(192, 192, 160), allow_missing_keys=True),
        # transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)

brats_shape_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys="seg", allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys="seg", allow_missing_keys=True),
        transforms.Orientationd(keys="seg", axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys="seg", source_key="seg", allow_missing_keys=True),
        transforms.CenterSpatialCropd(keys="seg", roi_size=(192, 192, 160)),
        transforms.SpatialPadd(keys="seg", spatial_size=(192, 192, 160), allow_missing_keys=True),
    ]
)

bratsMD_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1", "t1ce", "t2", "flair"]),
        transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"]),
        transforms.Lambdad(keys=["t1", "t1ce", "t2", "flair"], func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["t1", "t1ce", "t2", "flair"]),
        transforms.EnsureTyped(keys=["t1", "t1ce", "t2", "flair"]),
        transforms.Orientationd(keys=["t1", "t1ce", "t2", "flair"], axcodes="RAI"),
        transforms.CenterSpatialCropd(keys=["t1", "t1ce", "t2", "flair"], roi_size=(192, 192, 160)),
        transforms.SpatialPadd(keys=["t1", "t1ce", "t2", "flair"], spatial_size=(192, 192, 160), allow_missing_keys=True),
        transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t1ce", "t2", "flair"], lower=0, upper=100, b_min=0, b_max=1),
    ]
)

brats_aug_transforms = transforms.Compose(
    [
        transforms.RandRotate90d(keys="seg", prob=0.2, spatial_axes=(0,1)),
        transforms.RandRotate90d(keys="seg", prob=0.2, spatial_axes=(0,2)),
        transforms.RandRotate90d(keys="seg", prob=0.2, spatial_axes=(1,2)),
        # transforms.RandRotated(keys="seg", prob=0.4, range_x=0.3, range_y=0.3, range_z=0.3, mode='nearest'),
        transforms.RandAxisFlipd(keys="seg", prob=0.3),
    ]
)

def get_brats_dataset(data_path, radiomics_path):
    transform = brats_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    for i, subject in enumerate(os.listdir(data_path)):
        # if not (subject == "BraTS2021_01025"): continue 
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == subject].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        # t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz")
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz")
        seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")
        # t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz")
        # seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")
        data.append({"image":t1ce, "seg":seg, 'radiomics': radiomics})
        # data.append({"image":t1ce, "seg":seg, 'radiomics': np.array([0], dtype=np.float32)})
    # subject == "BraTS2021_01287"
    # image_sub = "BraTS2021_00306"
    # seg_sub = "BraTS2021_01094"
    # t1ce = os.path.join(data_path, image_sub, f"{image_sub}_t1ce.nii.gz")
    # seg = os.path.join(data_path, seg_sub, f"{seg_sub}_seg.nii.gz")
    # data.append({"image":t1ce, "seg":seg, 'radiomics': radiomics})
    for i in range(len(data)):
        data[i]['next_seg'] = data[(i+1) % len(data)]['seg']
        data[i]['next_radiomics'] = data[(i+1) % len(data)]['radiomics']
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

def get_brats_dataset_shape(data_path, radiomics_path):
    transform = brats_shape_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    for i, subject in enumerate(os.listdir(data_path)):
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == subject].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")

        data.append({"seg":seg, 'radiomics': radiomics})
                    
    print("num of subject:", len(data))
    return MonaiDataset(data=data, transform=transform)

def get_brats_multimodal(data_path):
    transform = bratsMD_transforms
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz")
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz")
        t2 = os.path.join(sub_path, f"{subject}_t2.nii.gz")
        flair = os.path.join(sub_path, f"{subject}_flair.nii.gz")

        data.append({"t1":t1, "t1ce":t1ce, "t2":t2, "flair":flair})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

def get_hcp_dataset(data_path):
    transform = brats_transforms

    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject, "T1w")
        if os.path.exists(sub_path) == False: continue
        img = os.path.join(sub_path, f"T1w_acpc_dc_restore_brain.nii.gz")

        data.append({"image":img})

    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

def get_ixi_multimodal(data_path):
    transform = brats_transforms

    data = []
    for subject in os.listdir(data_path):
        t1 = os.path.join(data_path, f"{subject}")
        if os.path.exists(t1) == False: continue

        data.append({"image": t1})
    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

# ---3D lung dataset ---

lung_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "seg"]),
        transforms.EnsureChannelFirstd(keys=["image", "seg"]),
        transforms.Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image", "seg"], axcodes="RAI"),
        # transforms.CenterSpatialCropd(keys=["image", "seg"], roi_size=(224, 224, 96)),
        # transforms.SpatialPadd(keys=["image", "seg"], spatial_size=(224, 224, 96), allow_missing_keys=True),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        transforms.CropForegroundd(keys=["image", "seg"], source_key="image"),
    ]
)

lung_shape_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys="seg", allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys="seg", allow_missing_keys=True),
        transforms.Orientationd(keys="seg", axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys="seg", source_key="seg", allow_missing_keys=True),
        transforms.CenterSpatialCropd(keys="seg", roi_size=(224, 224, 96)),
        transforms.SpatialPadd(keys="seg", spatial_size=(224, 224, 96), allow_missing_keys=True),
    ]
)

def get_lung_dataset(data_path, radiomics_path):
    transform = lung_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    for subject in os.listdir(data_path):
        if subject != "LUNG1-182": continue
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == subject].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        # img = os.path.join(sub_path, f"img-384.nii.gz")
        # seg = os.path.join(sub_path, f"tumor-384.nii.gz")

        img = os.path.join(sub_path, f"crop_img.nii.gz")
        seg = os.path.join(data_path, "LUNG1-314", f"crop_tumor.nii.gz")
        # img = os.path.join(sub_path, f"crop_img.nii.gz")
        # seg = os.path.join(sub_path, f"crop_tumor.nii.gz")

        data.append({"image":img, "seg":seg, 'radiomics': radiomics})
        # data.append({"image":img, "seg":seg, 'radiomics': np.array([0], dtype=np.float32)})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

def get_lung_shape_dataset(data_path, radiomics_path):
    transform = lung_shape_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == subject].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        seg = os.path.join(sub_path, f"crop_tumor.nii.gz")

        data.append({"seg":seg, 'radiomics': radiomics})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

@Registers.datasets.register_with_name('brats')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_brats_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)
        # self.sample_imgs = get_hcp_dataset(dataset_config.sample_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask == 1] = 1
        # mask[mask == 4] = 1
        mask[mask != 1] = 0
        cx, cy, cz = center_of_mass(mask[0].numpy())
        mx, my, mz = mask.shape[1], mask.shape[2], mask.shape[3]
        if np.isnan(cx): cx = 96
        if np.isnan(cy): cy = 96
        if np.isnan(cz): cz = 80
        cx, cy, cz = int(cx), int(cy), int(cz)

        # print(cx, cy, cz)
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

    # def tumor_crop(self, x, x_cond, x_seg):
    #     # sitk.WriteImage(sitk.GetImageFromArray(x[0]), "x.nii.gz")
    #     # sitk.WriteImage(sitk.GetImageFromArray(x_seg[0]), "x_seg.nii.gz")
    #     x_seg_clone = x_seg.clone()
    #     x_seg_clone[x != 0] = 1
    #     x_seg_clone[x_seg_clone != 1] = 0
    #     # x_seg_clone[x_seg == 1] = 3
    #     # x_seg_clone[x_seg == 4] = 2
    #     sitk.WriteImage(sitk.GetImageFromArray(x_seg_clone[0]), "x_whole.nii.gz")
    #     cx, cy, cz = 101, 91, 155-96
    #     sx, sy, sz, ex, ey, ez = cx - 56, cy - 56, cz - 48, cx + 56, cy + 56, cz + 48
    #     # sx, sy, sz, ex, ey, ez = 100-56, 156-56, 81-48, 100+56, 156+56, 81+48 #142-56, 115-56, 81-48, 142+56, 115+56, 81+48
    #     x = x[:,sx:ex, sy:ey, sz:ez]
    #     x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
    #     x_seg_clone = x_seg_clone[:,sx:ex, sy:ey, sz:ez]
        
    #     # x_seg = rearrange(x_seg, "b h w d -> b d h w")
    #     x_seg = zoom(x_seg, (1, 0.85, 0.85, 0.85), order=0)
    #     x_seg = torch.tensor(x_seg, dtype=torch.float32)
    #     sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)

    #     x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
    #     x_cond[x_seg == 1] = 0.5
    #     x_cond[x_seg == 4] = 1
    #     # x_cond[x_seg == 2] = 1
    #     x_seg_clone[x_seg == 1] = 3
    #     x_seg_clone[x_seg == 4] = 2
    #     # sitk.WriteImage(sitk.GetImageFromArray(x_seg_clone[0]), "x_seg_clone.nii.gz")

    #     # x = x[:,sx:ex, sy:ey, sz:ez]
    #     # x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
    #     # x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

    #     return x, x_cond, x_seg
    def tumor_crop(self, x, x_cond, x_seg, x_next_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)

        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

        nsx, nsy, nsz, nex, ney, nez = self.get_center_pos(x_next_seg)
        x_next_seg = x_next_seg[:,nsx:nex, nsy:ney, nsz:nez]
        
        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]

        x_cond[x_next_seg == 1] = 0.5
        x_cond[x_next_seg == 2] = 1

        pos = {"sx": sx, "sy": sy, "sz": sz, "ex": ex, "ey": ey, "ez": ez}

        return x, x_cond, x_next_seg, pos
    
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

        # x_image = sitk.GetImageFromArray(x[0])
        # sitk.WriteImage(x_image, "x.nii.gz")
        # x_cond_image = sitk.GetImageFromArray(x_cond[0])
        # sitk.WriteImage(x_cond_image, "x_cond.nii.gz")
        # x_seg_image = sitk.GetImageFromArray(x_seg[0])
        # sitk.WriteImage(x_seg_image, "x_seg.nii.gz")
        
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
            # x_cond[x_seg == 1] = 0.5
            # x_cond[x_seg == 4] = 1
            # x, x_cond, x_seg = self.tumor_crop(x, x_cond, x_seg)
            # x, x_cond = self.random_rotate_flip(x, x_cond)
            # x_rad = torch.tensor(np.array([0], dtype=np.float32))
            x_next_seg = self.imgs[i]['next_seg']
            x_next_rad = self.imgs[i]['next_radiomics']

            x_crop, x_cond_crop, x_seg_crop, pos = self.tumor_crop(x.clone(), x_cond.clone(), x_seg.clone(), x_next_seg.clone())
            x_cond[x_seg == 1] = 0.5
            x_cond[x_seg == 4] = 1
            subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-1].split('_')[1]
            return {"x": x, "x_cond": x_cond, "x_seg": x_seg, "x_next_seg": x_next_seg,
                    "x_crop": x_crop, "x_cond_crop": x_cond_crop, "x_seg_crop": x_seg_crop, "pos": pos,
                    "radiomics": x_rad, "x_next_rad": x_next_rad, "subject_id": subject_id}

        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-1].split('_')[1]
        return {"x": x , "x_cond": x_cond, "x_seg":x_seg, "radiomics": x_rad, "subject_id": subject_id}

@Registers.datasets.register_with_name('brats_change_shape')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_brats_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)
        # self.sample_imgs = get_hcp_dataset(dataset_config.sample_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask == 1] = 1
        mask[mask == 4] = 1
        mask[mask != 1] = 0
        cx, cy, cz = center_of_mass(mask[0])
        if np.isnan(cx): cx = 96
        if np.isnan(cy): cy = 96
        if np.isnan(cz): cz = 80
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 56, cy - 56, cz - 48
        ex, ey, ez = cx + 56, cy + 56, cz + 48
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 96
        if ex > 192: sx, ex = 80, 192
        if ey > 192: sy, ey = 80, 192
        if ez > 160: sz, ez = 64, 160
        
        return sx, sy, sz, ex, ey, ez
    
    def tumor_crop(self, x, x_cond, x_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_cond[x_seg == 1] = 0.5
        # x_cond[x_seg == 2] = 1 # edema
        x_cond[x_seg == 4] = 1

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        x_seg[x_seg == 2] = 0
        x_seg[x_seg == 1] = 0.5
        x_seg[x_seg == 4] = 1

        return x, x_cond, x_seg

    def __getitem__(self, i):
        x = self.imgs[i]['image']
        x_cond = self.imgs[i]['image']
        x_seg = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']
        
        x, x_cond, x_seg = self.tumor_crop(x, x_cond, x_seg)

        subject_id = self.imgs[i][f"image_meta_dict"]['filename_or_obj'].split('/')[-1].split('_')[1]
        return {"x": x , "x_cond": x_cond, "x_seg":x_seg, "radiomics": x_rad, "subject_id": subject_id}
    

@Registers.datasets.register_with_name('brats_shape')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_brats_dataset_shape(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask == 1] = 1
        mask[mask == 4] = 1
        mask[mask != 1] = 0
        cx, cy, cz = center_of_mass(mask[0])
        if np.isnan(cx): cx = 96
        if np.isnan(cy): cy = 96
        if np.isnan(cz): cz = 80
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 48, cy - 48, cz - 48
        ex, ey, ez = cx + 48, cy + 48, cz + 48
        if sx < 0: sx, ex = 0, 96
        if sy < 0: sy, ey = 0, 96
        if sz < 0: sz, ez = 0, 96
        if ex > 192: sx, ex = 96, 192
        if ey > 192: sy, ey = 96, 192
        if ez > 160: sz, ez = 64, 160
        
        return sx, sy, sz, ex, ey, ez
    
    def tumor_crop(self, x_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        x_seg[x_seg == 2] = 0
        x_seg[x_seg == 1] = 1
        x_seg[x_seg == 4] = 0.5

        return x_seg
    
    def get_low_res(self, x_seg):
        x_seg = x_seg[:, None]
        x = F.interpolate(x_seg, (24,24,24))
        x = F.interpolate(x, (96,96,96))
        x = x[:,0]
        
        return x

    def __getitem__(self, i):
        x = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']

        x = self.tumor_crop(x)
        # x = brats_aug_transforms({"seg": x})["seg"]
        x_cond = self.get_low_res(x)

        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-1].split('_')[1]
        return {"x": x , "x_cond": x_cond, "radiomics": x_rad, "subject_id": subject_id}

@Registers.datasets.register_with_name('brats_multimodal')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.imgs = get_brats_multimodal(dataset_config.dataset_path)
        self.modalities = ["t1", "t1ce", "t2", "flair"]

    def __len__(self):
        return len(self.imgs)
    
    def get_random_int(self):
        randint = np.random.randint(0, 4, size=2)
        while randint[0] == randint[1]:
            randint = np.random.randint(0, 4, size=2)
        return randint
    
    def one_hot_encoding(self, num):
        one_hot = np.zeros(4)
        one_hot[num] = 1
        return np.array(one_hot, dtype=np.float32)

    def __getitem__(self, i):
        src, tgt = self.get_random_int()
        src_mod, tgt_mod = self.modalities[src], self.modalities[tgt]
        x = self.imgs[i][tgt_mod]
        x_cond = self.imgs[i][src_mod]

        subject_id = self.imgs[i][f"image_meta_dict"]['filename_or_obj'].split('/')[-1].split('_')[1]
        
        return {"x": x , "x_cond": x_cond, "radiomics": self.one_hot_encoding(tgt), subject_id: subject_id}
    

@Registers.datasets.register_with_name('lung')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        # print(dataset_config.dataset_path, dataset_config.radiomics_path)
        self.imgs = get_lung_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, mask):
        mask = mask.clone()
        mask[mask != 0] = 1
        cx, cy, cz = center_of_mass(mask[0].numpy())
        mx, my, mz = mask.shape[1], mask.shape[2], mask.shape[3]
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 64, cy - 64, cz - 32
        ex, ey, ez = cx + 64, cy + 64, cz + 32
        if sx < 0: sx, ex = 0, 128
        if sy < 0: sy, ey = 0, 128
        if sz < 0: sz, ez = 0, 64
        if ex > mx: sx, ex = mx-128, mx
        if ey > my: sy, ey = my-128, my
        if ez > mz: sz, ez = mz-64, mz
        
        return sx, sy, sz, ex, ey, ez
    
    def tumor_crop(self, x, x_cond, x_seg):
        cx, cy, cz = 112, 112, 40
        sx, sy, sz, ex, ey, ez = cx - 64, cy - 64, cz - 32, cx + 64, cy + 64, cz + 32
        # sx, sy, sz, ex, ey, ez = 177-56, 222-56, 50-40, 177+56, 222+56, 50+40
        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]

        x_seg = zoom(x_seg, (1, 1.01, 1.01, 1.25), order=0)
        x_seg = torch.tensor(x_seg, dtype=torch.float32)
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)

        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        x_seg[:,:,:,-2:] = x_seg[:,:,:,:2]
        x_seg[:,:,:,:2] = 0
        x_seg[:,:,:10,:] = x_seg[:,:,:10,:]
        x_seg[:,:,:10,:] = 0
        # x_seg[:,:,40:108] = x_seg[:,:,60:128]
        # x_seg[:,:,108:] = 0
        x_cond[x_seg == 1] = 1
        # x = x[:,sx:ex, sy:ey, sz:ez]
        # x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        # x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

        return x, x_cond, x_seg
    
    def non_tumor_crop(self, x, x_cond, x_seg):
        tumor_mask = (x_seg[0] == 1)
        brain_mask = (x[0] > 0)
        
        distance_map = distance_transform_edt(np.logical_and(brain_mask, ~tumor_mask))
        cz, cy, cx = np.unravel_index(distance_map.argmax(), distance_map.shape)
        
        sx, sy, sz = cx - 56, cy - 56, cz - 40
        ex, ey, ez = cx + 56, cy + 56, cz + 40
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 80
        if ex > 224: sx, ex = 112, 224
        if ey > 224: sy, ey = 112, 224
        if ez > 96: sz, ez = 16, 96

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

        x_cond[x_seg == 1] = 1
        
        return x, x_cond
    
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

    def __getitem__(self, i):
        x = self.imgs[i]['image']
        x_cond = self.imgs[i]['image']
        x_seg = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']

        if x_rad.shape[0] == 1:
            x, x_cond = self.non_tumor_crop(x, x_cond, x_seg)
            x, x_cond = self.random_rotate_flip(x, x_cond)
        else:
            # x_cond[x_seg == 1] = 1
            x, x_cond, x_seg = self.tumor_crop(x, x_cond, x_seg)
            # x, x_cond = self.random_rotate_flip(x, x_cond)
            # x_rad = torch.tensor(np.array([0], dtype=np.float32))
        # print(x_cond.shape, x.shape)
            

        subject_id = self.imgs[i][f"image_meta_dict"]['filename_or_obj'].split('/')[-2]
        return {"x": x , "x_cond": x_cond, "x_seg":x_seg,"radiomics": x_rad, "subject_id": subject_id}
    
@Registers.datasets.register_with_name('lung_change_shape')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        # print(dataset_config.dataset_path, dataset_config.radiomics_path)
        self.imgs = get_lung_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, mask):
        mask[mask != 0] = 1
        cx, cy, cz = center_of_mass(mask[0])
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 56, cy - 56, cz - 40
        ex, ey, ez = cx + 56, cy + 56, cz + 40
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 80
        if ex > 224: sx, ex = 112, 224
        if ey > 224: sy, ey = 112, 224
        if ez > 96: sz, ez = 16, 96
        
        return sx, sy, sz, ex, ey, ez
    
    def tumor_crop(self, x, x_cond, x_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_cond[x_seg == 1] = 1

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

        x_seg[x_seg == 1] = 1
        x_seg[x_seg != 1] = 0
              
        return x, x_cond, x_seg

    def __getitem__(self, i):
        x = self.imgs[i]['image']
        x_cond = self.imgs[i]['image']
        x_seg = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']

        x, x_cond, x_seg = self.tumor_crop(x, x_cond, x_seg)

        subject_id = self.imgs[i][f"image_meta_dict"]['filename_or_obj'].split('/')[-2]
        return {"x": x , "x_cond": x_cond, "x_seg":x_seg, "radiomics": x_rad, "subject_id": subject_id}

@Registers.datasets.register_with_name('lung_shape')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_lung_shape_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask == 1] = 1
        mask[mask != 1] = 0
        cx, cy, cz = center_of_mass(mask[0])
        if np.isnan(cx): cx = 96
        if np.isnan(cy): cy = 96
        if np.isnan(cz): cz = 80
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 48, cy - 48, cz - 48
        ex, ey, ez = cx + 48, cy + 48, cz + 48
        if sx < 0: sx, ex = 0, 96
        if sy < 0: sy, ey = 0, 96
        if sz < 0: sz, ez = 0, 96
        if ex > 224: sx, ex = 128, 224
        if ey > 224: sy, ey = 128, 224
        if ez > 96: sz, ez = 0, 96
        
        return sx, sy, sz, ex, ey, ez
    
    def tumor_crop(self, x_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        x_seg[x_seg == 1] = 1

        return x_seg
    
    def get_low_res(self, x_seg):
        x_seg = x_seg[:, None]
        x = F.interpolate(x_seg, (24,24,24))
        x = F.interpolate(x, (96,96,96))
        x = x[:,0]
        
        return x

    def __getitem__(self, i):
        x = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']

        x = self.tumor_crop(x)
        # x = brats_aug_transforms({"seg": x})["seg"]
        x_cond = self.get_low_res(x)

        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-2]
        return {"x": x , "x_cond": x_cond, "radiomics": x_rad, "subject_id": subject_id}
    

# breast_transforms = transforms.Compose(
#     [
#         transforms.LoadImaged(keys=["image", "seg"], allow_missing_keys=True),
#         transforms.EnsureChannelFirstd(keys=["image", "seg"]),
#         transforms.Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
#         transforms.AddChanneld(keys=["image"]),
#         transforms.EnsureTyped(keys=["image"]),
#         transforms.Orientationd(keys=["image", "seg"], axcodes="RAI"),
#         transforms.CropForegroundd(keys=["image", "seg"], source_key="image", allow_missing_keys=True),
#         transforms.CenterSpatialCropd(keys=["image", "seg"], roi_size=(192, 192, 160)),
#         transforms.SpatialPadd(keys=["image", "seg"], spatial_size=(192, 192, 160)),
#         transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
#     ]
# )

breast_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "seg", "next_seg"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "seg", "next_seg"]),
        transforms.Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image", "seg", "next_seg"], axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys=["image", "seg", "next_seg"], source_key="image", allow_missing_keys=True),
        transforms.CenterSpatialCropd(keys=["image", "seg", "next_seg"], roi_size=(192, 192, 160), allow_missing_keys=True),
        transforms.SpatialPadd(keys=["image", "seg", "next_seg"], spatial_size=(192, 192, 160), allow_missing_keys=True),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)

def get_breast_dataset(data_path, radiomics_path):
    transform = breast_transforms
    radiomics_df = pd.read_csv(radiomics_path)

    data = []
    for i, subject in enumerate(os.listdir(data_path)):
        # if not (subject == "60"): continue
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == int(subject)].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        
        # eth = os.path.join(sub_path, "eth2-256.nii.gz")
        # seg = os.path.join(sub_path, "roi-256.nii.gz")
        # data.append({"image":eth, "seg":seg, 'radiomics': radiomics})

        eth_l = os.path.join(sub_path, "left", "eth2.nii.gz")
        eth_r = os.path.join(sub_path, "right", "eth2.nii.gz")
        seg_l = os.path.join(sub_path, "left", "roi.nii.gz")
        seg_r = os.path.join(sub_path, "right", "roi.nii.gz")

        # roi = os.path.join(data_path, "37", "left", "roi.nii.gz")

        # data.append({"image":eth_l, "seg":roi, 'radiomics': radiomics})
        if os.path.exists(seg_l):
            data.append({"image":eth_l, "seg":seg_l, 'radiomics': radiomics})
            # data.append({"image":eth_r, "seg":seg_l, 'radiomics': np.array([0], dtype=np.float32)})
        elif os.path.exists(seg_r):
            data.append({"image":eth_r, "seg":seg_r, 'radiomics': radiomics})
            # data.append({"image":eth_l, "seg":seg_r, 'radiomics': np.array([0], dtype=np.float32)})

    for i in range(len(data)):
        data[i]["next_seg"] = data[(i+1)%len(data)]["seg"]
        data[i]["next_radiomics"] = data[(i+1)%len(data)]["radiomics"]
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

@Registers.datasets.register_with_name('breast')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_breast_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask == 1] = 1
        mask[mask == 2] = 1
        mask[mask != 1] = 0
        mx, my, mz = mask.shape[1], mask.shape[2], mask.shape[3]
        cx, cy, cz = center_of_mass(mask[0].numpy())
        if np.isnan(cx): cx = 96
        if np.isnan(cy): cy = 96
        if np.isnan(cz): cz = 80
        cx, cy, cz = int(cx), int(cy), int(cz)
        # print(cx, cy, cz)
        sx, sy, sz = cx - 56, cy - 56, cz - 40
        ex, ey, ez = cx + 56, cy + 56, cz + 40
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 80
        if ex > mx: sx, ex = mx-112, mx
        if ey > my: sy, ey = my-112, my
        if ez > mz: sz, ez = mz-80, mz
        
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
    
    # def tumor_crop(self, x, x_cond, x_seg):
    def tumor_crop(self, x, x_cond, x_seg, x_next_seg):
        # sx, sy, sz, ex, ey, ez = 98-56, 120-56, 82-40, 98+56, 120+56, 82+40
        # x = x[:,sx:ex, sy:ey, sz:ez]
        # x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        # x_seg = zoom(x_seg, (1, 1.1, 0.9, 1.1), order=0)
        # print(x_seg.shape, x.shape)
        # x_seg = torch.tensor(np.array(x_seg)).to(x.device)

        # x_seg[:,:,50-20:112-20] = x_seg[:,:,50:112]
        # x_seg[:,:,112-20:] = 0

        # return x, x_cond, x_seg
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)

        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

        nsx, nsy, nsz, nex, ney, nez = self.get_center_pos(x_next_seg)
        x_next_seg = x_next_seg[:,nsx:nex, nsy:ney, nsz:nez]

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]

        x_cond[x_next_seg == 1] = 0.5
        x_cond[x_next_seg == 2] = 1

        pos = {"sx": sx, "sy": sy, "sz": sz, "ex": ex, "ey": ey, "ez": ez}

        return x, x_cond, x_next_seg, pos
    
    def non_tumor_crop(self, x, x_cond, x_seg):
        tumor_mask = (x_seg[0] == 1) | (x_seg[0] == 2)
        brain_mask = (x[0] > 0)

        mx, my, mz = x_seg.shape[1], x_seg.shape[2], x_seg.shape[3]
        distance_map = distance_transform_edt(np.logical_and(brain_mask, ~tumor_mask))
        cz, cy, cx = np.unravel_index(distance_map.argmax(), distance_map.shape)
        
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 56, cy - 56, cz - 40
        ex, ey, ez = cx + 56, cy + 56, cz + 40
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 80
        if ex > mx: sx, ex = mx-112, mx
        if ey > my: sy, ey = my-112, my
        if ez > mz: sz, ez = mz-80, mz

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

        x_cond[x_seg == 1] = 0.5
        x_cond[x_seg == 2] = 1
        
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
            x_next_seg = self.imgs[i]['next_seg']
            x_next_rad = self.imgs[i]['next_radiomics']
            # patch_x, _ = self.tumor_crop(x, x_cond, x_seg)
            # patch_x_min = patch_x.min()
            # patch_x_max = torch.quantile(patch_x, 0.995)
            # # print(patch_x_max, patch_x_min)
            # x = (x - patch_x_min) / (patch_x_max - patch_x_min)
            # x_cond = (x_cond - patch_x_min) / (patch_x_max - patch_x_min)
            # x_cond[x_seg == 1] = 0.5
            # x_cond[x_seg == 2] = 1
            # x, x_cond, x_seg = self.tumor_crop(x, x_cond, x_seg)
            x_crop, x_cond_crop, x_seg_crop, pos = self.tumor_crop(x.clone(), x_cond.clone(), x_seg.clone(), x_next_seg.clone())
            x_cond[x_seg == 1] = 0.5
            x_cond[x_seg == 2] = 1
            subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-3]
            return {"x": x, "x_cond": x_cond, "x_seg": x_seg, "x_next_seg": x_next_seg,
                    "x_crop": x_crop, "x_cond_crop": x_cond_crop, "x_seg_crop": x_seg_crop, "pos": pos,
                    "radiomics": x_rad, "x_next_rad": x_next_rad, "subject_id": subject_id}    
            # x, x_cond = self.random_rotate_flip(x, x_cond)
            # x_rad = torch.tensor(np.array([0], dtype=np.float32))

        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-3]
        return {"x": x , "x_cond": x_cond, "x_seg": x_seg, "radiomics": x_rad, "subject_id": subject_id}
    
@Registers.datasets.register_with_name('breast_change_shape')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_breast_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask == 1] = 1
        mask[mask == 2] = 1
        mask[mask != 1] = 0
        mx, my, mz = mask.shape[1], mask.shape[2], mask.shape[3]
        cx, cy, cz = center_of_mass(mask[0])
        if np.isnan(cx): cx = 96
        if np.isnan(cy): cy = 96
        if np.isnan(cz): cz = 80
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 56, cy - 56, cz - 40
        ex, ey, ez = cx + 56, cy + 56, cz + 40
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 80
        if ex > mx: sx, ex = mx-112, mx
        if ey > my: sy, ey = my-112, my
        if ez > mz: sz, ez = mz-80, mz
        
        return sx, sy, sz, ex, ey, ez
    
    def tumor_crop(self, x, x_cond, x_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_cond[x_seg == 1] = 0.5
        x_cond[x_seg == 2] = 1

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

        x_seg[x_seg == 1] = 0.5
        x_seg[x_seg == 2] = 1

        return x, x_cond, x_seg

    def __getitem__(self, i):
        x = self.imgs[i]['image']
        x_cond = self.imgs[i]['image']
        x_seg = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']
    
        x, x_cond, x_seg = self.tumor_crop(x, x_cond, x_seg)

        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-3]
        return {"x": x , "x_cond": x_cond, "x_seg":x_seg, "radiomics": x_rad, "subject_id": subject_id}
    
breast_shape_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys="seg", allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys="seg", allow_missing_keys=True),
        transforms.Orientationd(keys="seg", axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys="seg", source_key="seg", allow_missing_keys=True),
        transforms.CenterSpatialCropd(keys="seg", roi_size=(192, 192, 160)),
        transforms.SpatialPadd(keys="seg", spatial_size=(192, 192, 160), allow_missing_keys=True),
    ]
)

def get_breast_dataset_shape(data_path, radiomics_path):
    transform = breast_shape_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    for i, subject in enumerate(os.listdir(data_path)):
        sub_path = os.path.join(data_path, subject)
        try: radiomics = radiomics_df[radiomics_df["subject"] == int(subject)].values[0][1:].astype(np.float32)
        except: continue
        if os.path.exists(sub_path) == False: continue
        seg = os.path.join(sub_path, "roi.nii.gz")

        data.append({"seg":seg, 'radiomics': radiomics})
                    
    print("num of subject:", len(data))
    return MonaiDataset(data=data, transform=transform)


@Registers.datasets.register_with_name('breast_shape')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_breast_dataset_shape(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask == 1] = 1
        mask[mask == 2] = 1
        mask[mask != 1] = 0
        cx, cy, cz = center_of_mass(mask[0])
        if np.isnan(cx): cx = 96
        if np.isnan(cy): cy = 96
        if np.isnan(cz): cz = 80
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 48, cy - 48, cz - 48
        ex, ey, ez = cx + 48, cy + 48, cz + 48
        if sx < 0: sx, ex = 0, 96
        if sy < 0: sy, ey = 0, 96
        if sz < 0: sz, ez = 0, 96
        if ex > 192: sx, ex = 96, 192
        if ey > 192: sy, ey = 96, 192
        if ez > 160: sz, ez = 64, 160
        
        return sx, sy, sz, ex, ey, ez
    
    def tumor_crop(self, x_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        x_seg[x_seg == 1] = 0.5
        x_seg[x_seg == 2] = 1

        return x_seg
    
    def get_low_res(self, x_seg):
        x_seg = x_seg[:, None]
        x = F.interpolate(x_seg, (24,24,24))
        x = F.interpolate(x, (96,96,96))
        x = x[:,0]
        
        return x

    def __getitem__(self, i):
        x = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']

        x = self.tumor_crop(x)
        # x = brats_aug_transforms({"seg": x})["seg"]
        x_cond = self.get_low_res(x)

        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-2]
        return {"x": x , "x_cond": x_cond, "radiomics": x_rad, "subject_id": subject_id}
    


kit23_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "seg"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "seg"]),
        transforms.Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image", "seg"], axcodes="SAR"),
        # transforms.CropForegroundd(keys=["image", "seg"], source_key="image", allow_missing_keys=True),
        # transforms.CenterSpatialCropd(keys=["image", "seg"], roi_size=(192, 192, 160)),
        # transforms.SpatialPadd(keys=["image", "seg"], spatial_size=(192, 192, 160)),
        transforms.CropForegroundd(keys=["seg"], source_key="seg", allow_missing_keys=True),
        transforms.CenterSpatialCropd(keys=["seg"], roi_size=(160, 192, 192)),
        transforms.SpatialPadd(keys=["seg"], spatial_size=(160, 192, 192)),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        # transforms.CropForegroundd(keys=["image", "seg"], source_key="image"),
    ]
)

def get_kit23_dataset(data_path, radiomics_path):
    transform = kit23_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    phase = ['left', 'right']
    for i, subject in enumerate(os.listdir(data_path)):
        if subject != "case_00226": continue
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        for p in phase:
            try: radiomics =  radiomics_df[(radiomics_df["subject"] == subject) & (radiomics_df["phase"] == p)].values[0][2:].astype(np.float32)
            except: continue
            # ct = os.path.join(sub_path, f"imaging-384.nii.gz")
            # seg = os.path.join(sub_path, f"segmentation-384.nii.gz")

            ct = os.path.join(sub_path, f"imaging.nii.gz")
            seg = os.path.join(data_path, "case_00419", f"right_crop_seg.nii.gz")
            # ct = os.path.join(sub_path, f"{p}_crop_img.nii.gz")
            # seg = os.path.join(sub_path, f"{p}_crop_seg.nii.gz")

            if not os.path.exists(seg): continue

            data.append({"image":ct, "seg":seg, 'radiomics': radiomics})
            # data.append({"image":ct, "seg":seg, 'radiomics': np.array([0], dtype=np.float32)})
        # ct = os.path.join(sub_path, f"imaging.nii.gz")
        # kid = os.path.join(sub_path, f"right_seg.nii.gz")
        # seg = os.path.join(data_path, "case_00031", f"right_seg.nii.gz")
        # seg = os.path.join(data_path, "case_00133", f"left_seg.nii.gz")
        # seg = os.path.join(data_path, "case_00144", f"right_seg.nii.gz")
        # data.append({"image":ct, "seg":seg, 'kid':kid, 'radiomics': radiomics})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

@Registers.datasets.register_with_name('kidney')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_kit23_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask != 2] = 0
        mask[mask == 2] = 1
        mx, my, mz = mask.shape[1], mask.shape[2], mask.shape[3]
        if mz < 112:
            mask_clone = torch.zeros((1, mx, my, 112))
            cen = (112 - mz) // 2
            mask_clone[:, :, :, cen:cen+mz] = mask
            mask = mask_clone
        cx, cy, cz = center_of_mass(mask[0].numpy())
        print(cx, cy, cz)
        if np.isnan(cx): cx = mx//2
        if np.isnan(cy): cy = my//2
        if np.isnan(cz): cz = mz//2
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 56, cy - 56, cz - 56
        ex, ey, ez = cx + 56, cy + 56, cz + 56
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 112
        if ex > mx: sx, ex = mx-112, mx
        if ey > my: sy, ey = my-112, my
        if ez > mz: sz, ez = mz-112, mz
        
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
        # sitk.WriteImage(sitk.GetImageFromArray(x[0]), "x.nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(x_seg[0]), "x_seg.nii.gz")
        cx, cy, cz = 160-56, 512-278, 512-355
        sx, sy, sz, ex, ey, ez = cx - 56, cy - 56, cz - 56, cx + 56, cy + 56, cz + 56
        # sx, sy, sz, ex, ey, ez = 129-56, 198-56, 207-56, 129+56, 198+56, 207+56
        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        # x_kid = x_kid[:,sx:ex, sy:ey, sz:ez]
        x_seg = zoom(x_seg, (1, 3.25, 0.7, 0.7), order=0)
        x_seg = torch.tensor(np.array(x_seg)).to(x.device)
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        print(x_seg.shape)
        
        # x_seg_zoom = x_seg.clone().detach().cpu().numpy()
        # x_seg_zoom = zoom(x_seg_zoom, (1, 0.8, 0.8, 0.8), order=0)
        # x_seg_zoom = np.array(x_seg_zoom)

        # center_x, center_y, center_z = map(int, center_of_mass(x_seg_zoom[0]))
        # start_x, start_y, start_z = (
        #     center_x - x_seg_zoom.shape[1] // 2, 
        #     center_y - x_seg_zoom.shape[2] // 2, 
        #     center_z - x_seg_zoom.shape[3] // 2
        # )
        # end_x, end_y, end_z = (
        #     start_x + x_seg_zoom.shape[1], 
        #     start_y + x_seg_zoom.shape[2], 
        #     start_z + x_seg_zoom.shape[3]
        # )
        
        # x_seg_clone = np.zeros_like(x_seg.clone().detach().cpu().numpy())
        # print(start_x, end_x, start_y, end_y, start_z, end_z)
        # x_seg_clone[:,start_x:end_x, 0:90, start_z:end_z] = x_seg_zoom
        # x_seg = torch.tensor(np.array(x_seg_clone))
        # x_seg[:,0:22] = x_seg[:,23:45]
        # x_seg[:,23:45] = 0
        
        x_cond[x_seg == 2] = 1
        # x_kid[x_seg == 2] = 2
        # x_seg[x_seg == 1] = 0
        # x_seg[x_seg == 2] = 1
        # sitk.WriteImage(sitk.GetImageFromArray(x_kid[0]), "x_kid.nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(x_seg[0]), "x_seg.nii.gz")

        # x = x[:,sx:ex, sy:ey, sz:ez]
        # x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        # x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        return x, x_cond, x_seg
    
    def non_tumor_crop(self, x, x_cond, x_seg):
        tumor_mask = (x_seg[0] == 2)
        brain_mask = (x[0] > 0)

        mx, my, mz = x_seg.shape[1], x_seg.shape[2], x_seg.shape[3]
        if mz < 96:
            mask_clone = torch.zeros((1, mx, my, 96))
            cen = (96 - mz) // 2
            mask_clone[:, :, :, cen:cen+mz] = x_seg
            x_seg = mask_clone

        distance_map = distance_transform_edt(np.logical_and(brain_mask, ~tumor_mask))
        cz, cy, cx = np.unravel_index(distance_map.argmax(), distance_map.shape)
        
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 56, cy - 56, cz - 48
        ex, ey, ez = cx + 56, cy + 56, cz + 48
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 96
        if ex > mx: sx, ex = mx-112, mx
        if ey > my: sy, ey = my-112, my
        if ez > mz: sz, ez = mz-96, mz

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]

        x_cond[x_seg == 2] = 1
        
        return x, x_cond

    def __getitem__(self, i):
        x = self.imgs[i]['image']
        # cx, cy, cz = 61, 266, 345
        # x = x[:,:, cy-96:cy+96, cz-96:cz+96]
        # x_cond = x.clone()
        x_cond = self.imgs[i]['image']
        x_seg = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']
        # x_kid = self.imgs[i]['kid']
        
        if x_rad.shape[0] == 1:
            x, x_cond = self.non_tumor_crop(x, x_cond, x_seg)
            x, x_cond = self.random_rotate_flip(x, x_cond)
        else:
            x, x_cond, x_seg = self.tumor_crop(x, x_cond, x_seg)
            # x_cond[x_seg == 2] = 1
            # x, x_cond = self.random_rotate_flip(x, x_cond)
            # x_rad = torch.tensor(np.array([0], dtype=np.float32))


        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-2]
        return {"x": x , "x_cond": x_cond, "x_seg":x_seg, "radiomics": x_rad, "subject_id": subject_id}
    

@Registers.datasets.register_with_name('kidney_change_shape')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_kit23_dataset(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask != 2] = 0
        mask[mask == 2] = 1
        mx, my, mz = mask.shape[1], mask.shape[2], mask.shape[3]
        if mz < 96:
            mask_clone = torch.zeros((1, mx, my, 96))
            cen = (96 - mz) // 2
            mask_clone[:, :, :, cen:cen+mz] = mask
            mask = mask_clone
        cx, cy, cz = center_of_mass(mask[0])
        if np.isnan(cx): cx = mx//2
        if np.isnan(cy): cy = my//2
        if np.isnan(cz): cz = mz//2
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 56, cy - 56, cz - 56
        ex, ey, ez = cx + 56, cy + 56, cz + 56
        if sx < 0: sx, ex = 0, 112
        if sy < 0: sy, ey = 0, 112
        if sz < 0: sz, ez = 0, 112
        if ex > mx: sx, ex = mx-112, mx
        if ey > my: sy, ey = my-112, my
        if ez > mz: sz, ez = mz-112, mz
        
        return sx, sy, sz, ex, ey, ez
    
    def tumor_crop(self, x, x_cond, x_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_cond[x_seg == 2] = 1

        x = x[:,sx:ex, sy:ey, sz:ez]
        x_cond = x_cond[:,sx:ex, sy:ey, sz:ez]
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        
        x_seg[x_seg != 2] = 0
        x_seg[x_seg == 2] = 1

        return x, x_cond, x_seg

    def __getitem__(self, i):
        x = self.imgs[i]['image']
        x_cond = self.imgs[i]['image']
        x_seg = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']

        x, x_cond, x_seg = self.tumor_crop(x, x_cond, x_seg)

        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-2]
        return {"x": x , "x_cond": x_cond, "x_seg":x_seg, "radiomics": x_rad, "subject_id": subject_id}
    

def get_kit23_dataset_shape(data_path, radiomics_path):
    transform = breast_shape_transforms
    radiomics_df = pd.read_csv(radiomics_path)
    
    data = []
    phase = ['left', 'right']
    for i, subject in enumerate(os.listdir(data_path)):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        for p in phase:
            try: radiomics =  radiomics_df[(radiomics_df["subject"] == subject) & (radiomics_df["phase"] == p)].values[0][2:].astype(np.float32)
            except: continue
            seg = os.path.join(sub_path, f"{p}_seg.nii.gz")

            if not os.path.exists(seg): continue

            data.append({"seg":seg, 'radiomics': radiomics})

    print("num of subject:", len(data))
    return MonaiDataset(data=data, transform=transform)


@Registers.datasets.register_with_name('kidney_shape')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.imgs = get_kit23_dataset_shape(dataset_config.dataset_path, dataset_config.radiomics_path)

    def __len__(self):
        return len(self.imgs)
    
    def get_center_pos(self, seg):
        mask = seg.clone()
        mask[mask != 2] = 0
        mask[mask == 2] = 1
        mx, my, mz = mask.shape[1], mask.shape[2], mask.shape[3]
        cx, cy, cz = center_of_mass(mask[0])
        if np.isnan(cx): cx = mx//2
        if np.isnan(cy): cy = my//2
        if np.isnan(cz): cz = mz//2
        cx, cy, cz = int(cx), int(cy), int(cz)
        sx, sy, sz = cx - 32, cy - 32, cz - 32
        ex, ey, ez = cx + 32, cy + 32, cz + 32
        if sx < 0: sx, ex = 0, 64
        if sy < 0: sy, ey = 0, 64
        if sz < 0: sz, ez = 0, 64
        if ex > mx: sx, ex = mx-64, mx
        if ey > my: sy, ey = my-64, my
        if ez > mz: sz, ez = mz-64, mz
        
        return sx, sy, sz, ex, ey, ez
    
    def tumor_crop(self, x_seg):
        sx, sy, sz, ex, ey, ez = self.get_center_pos(x_seg)
        x_seg = x_seg[:,sx:ex, sy:ey, sz:ez]
        x_seg[x_seg != 2] = 0
        x_seg[x_seg == 2] = 1

        return x_seg
    
    def get_low_res(self, x_seg):
        x_seg = x_seg[:, None]
        x = F.interpolate(x_seg, (16,16,16))
        x = F.interpolate(x, (64,64,64))
        x = x[:,0]
        
        return x

    def __getitem__(self, i):
        x = self.imgs[i]['seg']
        x_rad = self.imgs[i]['radiomics']

        x = self.tumor_crop(x)
        # x = brats_aug_transforms({"seg": x})["seg"]
        x_cond = self.get_low_res(x)

        subject_id = self.imgs[i][f"seg_meta_dict"]['filename_or_obj'].split('/')[-2]
        return {"x": x , "x_cond": x_cond, "radiomics": x_rad, "subject_id": subject_id}