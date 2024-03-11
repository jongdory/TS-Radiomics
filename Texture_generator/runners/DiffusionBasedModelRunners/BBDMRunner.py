import os
import SimpleITK as sitk
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm
from torchsummary import summary

import numpy as np
import pandas as pd
from scipy.ndimage import measurements, zoom, rotate


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, _, _ = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            x, x_cond, x_rad, subject_id = batch['x'], batch['x_cond'], batch['radiomics'], batch['subject_id'][0]
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3, 4], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3, 4], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            x, x_cond, x_rad, subject_id = batch['x'], batch['x_cond'], batch['radiomics'], batch['subject_id'][0]
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3, 4], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3, 4], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        x, x_cond, x_rad, _ = batch['x'], batch['x_cond'], batch['radiomics'], batch['subject_id']
        x_rad = x_rad.unsqueeze(2)

        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])
        x_rad = x_rad.to(self.config.training.device[0])

        loss, additional_info = net(x, x_cond, context=x_rad)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))

        x, x_cond, x_rad, subject_id = batch['x'], batch['x_cond'], batch['radiomics'], batch['subject_id'][0]
        x_rad = x_rad.unsqueeze(2)

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])
        x_rad = x_rad[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        sample = net.sample(x_cond, context=x_rad, clip_denoised=self.config.testing.clip_denoised).to('cpu')
        sample[sample < 0] = 0
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        image_grid.savefig(os.path.join(sample_path, f'{subject_id}_skip_sample.png'))
        self.save_nifti(sample, os.path.join(sample_path, f'{subject_id}_skip_sample.nii.gz'))

        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        image_grid.savefig(os.path.join(sample_path, f'{subject_id}_condition.png'))
        self.save_nifti(x_cond, os.path.join(sample_path, f'{subject_id}_condition.nii.gz'))

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        image_grid.savefig(os.path.join(sample_path, f'{subject_id}_ground_truth.png'))
        self.save_nifti(x, os.path.join(sample_path, f'{subject_id}_ground_truth.nii.gz'))

        if self.config.model.model_type == "LBBDM":
            x_recon = net.decode(net.encode(x)).to('cpu')
            image_grid = get_image_grid(x_recon, grid_size, to_normal=self.config.data.dataset_config.to_normal)
            image_grid.savefig(os.path.join(sample_path, f'{subject_id}_x_recon.png'))

            x_cond_recon = net.decode(net.encode(x_cond, cond=True), cond=True).to('cpu')
            image_grid = get_image_grid(x_cond_recon, grid_size, to_normal=self.config.data.dataset_config.to_normal)
            image_grid.savefig(os.path.join(sample_path, f'{subject_id}_x_cond_recon.png'))
    
    @torch.no_grad()
    def save_nifti(self, sample, path):
        sample = sample.detach().cpu().numpy()[0,0]
        sample_image = sitk.GetImageFromArray(sample)
        sitk.WriteImage(sample_image, path)

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = 100
        for test_batch in pbar:
            if not 'change' in self.config.data.dataset_type:
                x, x_cond, x_rad, subject_id = test_batch['x'], test_batch['x_cond'], test_batch['radiomics'], test_batch['subject_id'][0]
            else:
                x, x_cond, x_seg, x_rad, subject_id = test_batch['x'], test_batch['x_cond'], test_batch['x_seg'], test_batch['radiomics'], test_batch['subject_id'][0]
            x_rad = x_rad.unsqueeze(2)

            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            x_rec = net.decode(net.encode(x)).to(self.config.training.device[0])
            x_rad = x_rad.to(self.config.training.device[0])