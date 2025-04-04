import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image
from datetime import datetime
from torchvision.utils import make_grid, save_image
from Register import Registers
from datasets.custom import CustomSingleDataset

def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir


def make_save_dirs(args, prefix, suffix=None, with_time=False):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if with_time else ""
    suffix = suffix if suffix is not None else ""

    result_path = make_dir(os.path.join(args.result_path, prefix, suffix, time_str))
    image_path = make_dir(os.path.join(result_path, "image"))
    log_path = make_dir(os.path.join(result_path, "log"))
    checkpoint_path = make_dir(os.path.join(result_path, "checkpoint"))
    sample_path = make_dir(os.path.join(result_path, "samples"))
    sample_to_eval_path = make_dir(os.path.join(result_path, "sample_to_eval"))
    print("create output path " + result_path)
    return image_path, checkpoint_path, log_path, sample_path, sample_to_eval_path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        return NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))


def get_dataset(data_config):
    train_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='train')
    val_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='val')
    test_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='test')
    return train_dataset, val_dataset, test_dataset


@torch.no_grad()
def save_single_image(image, save_path, file_name, to_normal=True):
    image = image.detach().clone()
    fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(15, 5))
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    image = image[0].cpu()
    axs[0].imshow(image[:, :, image.shape[2] // 2].rot90(1).detach().numpy(), cmap="gray")
    axs[1].imshow(image[:, image.shape[1] // 2, :].rot90(3).detach().numpy(), cmap="gray")
    axs[2].imshow(image[image.shape[0] // 2, :, :].rot90(3).detach().numpy(), cmap="gray")
    axs[0].set_ylabel(f"genetrated image")

    fig.savefig(os.path.join(save_path, f'{file_name}.png'))
    sitk.WriteImage(sitk.GetImageFromArray(image.cpu().numpy()), os.path.join(save_path, f'{file_name}.nii.gz'))

@torch.no_grad()
def get_image_grid(batch, grid_size=4, to_normal=True):
    batch = batch.detach().clone()
    fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(15, 5))
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    image = batch[0,0].cpu()
    axs[0].imshow(image[:, :, image.shape[2] // 2].rot90(1).detach().numpy(), cmap="gray")
    axs[1].imshow(image[:, image.shape[1] // 2, :].rot90(3).detach().numpy(), cmap="gray")
    axs[2].imshow(image[image.shape[0] // 2, :, :].rot90(3).detach().numpy(), cmap="gray")
    axs[0].set_ylabel(f"genetrated image")

    return fig