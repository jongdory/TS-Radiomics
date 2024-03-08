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
# from datasets.brats import get_brats_dataset


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
            # self.get_latent_mean_std()
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
        # train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        # train_dataset = get_brats_dataset(self.config.data.dataset_config.dataset_path, self.config.data.dataset_config.radiomics_path)
        train_dataset, _, _ = get_dataset(self.config.data) # get_brats_dataset()
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
            # (x, x_name), (x_cond, x_cond_name) = batch
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
            # (x, x_name), (x_cond, x_cond_name) = batch
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
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

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
        # reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        # reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

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
        if "shape" in self.config.data.dataset_type:
            if "lung" in self.config.data.dataset_type or "kidney" in self.config.data.dataset_type:
                sample[sample < 0.4] = 0
                sample[0.4 <= sample] = 1
            else:
                sample[sample < 0.35] = 0
                sample[(0.35 <= sample) & (sample <= 0.65)] = 0.5
                sample[sample > 0.65] = 1
            image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
            image_grid.savefig(os.path.join(sample_path, f'{subject_id}_skip_sample_p.png'))
            self.save_nifti(sample, os.path.join(sample_path, f'{subject_id}_skip_sample_p.nii.gz'))
        # im = Image.fromarray(image_grid)
        # im.save(os.path.join(sample_path, 'skip_sample.png'))
        # if stage != 'test':
        #     self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        image_grid.savefig(os.path.join(sample_path, f'{subject_id}_condition.png'))
        self.save_nifti(x_cond, os.path.join(sample_path, f'{subject_id}_condition.nii.gz'))
        # im = Image.fromarray(image_grid)
        # im.save(os.path.join(sample_path, 'condition.png'))
        # if stage != 'test':
        #     self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

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
        # im = Image.fromarray(image_grid)
        # im.save(os.path.join(sample_path, 'ground_truth.png'))
        # if stage != 'test':
        #     self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
    
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
        # sample_num = self.config.testing.sample_num
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

            # subject_id = test_batch["image_meta_dict"]['filename_or_obj'][0].split('/')[-1].split('_')[1]
            if 'change' in self.config.data.dataset_type:
                x_seg = x_seg.to(self.config.training.device[0])
                sub_sample_path = make_dir(os.path.join(sample_path, 'tumor_change_shape2',f'{subject_id}'))
                x_remove = net.sample(x_cond, context=torch.tensor([[0.]]).to(self.config.training.device[0]), clip_denoised=False)
                rotates = [(10,10,10), (10,-10,10), (-15,-25,-15), (-15,-15, 15), (20,20,20), (20,-15, 20)]
                scales = [1, 1.3, 1.5, 1.7, 1.9, 2.1] # [0.7, 0.85, 1, 1.15, 1.3, 1.45]
                condition = x_cond[0].detach().clone()
                gt = x[0]
                save_single_image(condition, sub_sample_path, f'condition', to_normal=to_normal)
                save_single_image(gt, sub_sample_path, f'groundtruth', to_normal=to_normal)
                save_single_image(x_remove[0], sub_sample_path, f'remove', to_normal=to_normal)
                recon = x_rec[0]
                save_single_image(recon, sub_sample_path, f'recon', to_normal=to_normal)

                radiomics_df = pd.read_csv(self.config.data.dataset_config.radiomics_path)
                max_rad = np.array(radiomics_df.max().iloc[2:]).astype(np.float32)
                min_rad = np.array(radiomics_df.min().iloc[2:]).astype(np.float32)
                max_rad = torch.tensor(max_rad).unsqueeze(0).unsqueeze(2).to(self.config.training.device[0])
                min_rad = torch.tensor(min_rad).unsqueeze(0).unsqueeze(2).to(self.config.training.device[0])
                zero_rad = torch.zeros_like(max_rad).to(self.config.training.device[0])
                min3_rad = -3 * torch.ones_like(max_rad).to(self.config.training.device[0])
                max3_rad = 3 * torch.ones_like(max_rad).to(self.config.training.device[0])
                rad_names = ["max", "min", "zero", "min3", "max3"]
                rads = [max_rad, min_rad, zero_rad, min3_rad, max3_rad]
                
                for j, rotate, scale in zip(range(6), rotates, scales):
                    x_masking = get_masking(x_remove, x_seg, scale, rotate)
                    if x_masking is None: continue
                    else: x_masking = x_masking.to(self.config.training.device[0])
                    save_single_image(x_masking[0], sub_sample_path, f'x_masking_{j}', to_normal=to_normal)

                    sample = net.sample(x_masking, context=x_rad, clip_denoised=False)
                    result = sample[0]
                    if sample_num > 1:
                        save_single_image(result, sub_sample_path, f'result_{j}', to_normal=to_normal)
                    else:
                        save_single_image(result, sub_sample_path, f'result', to_normal=to_normal)

                    # for rad_name, rad in zip(rad_names, rads):
                    #     result = net.sample(x_masking, context=rad, clip_denoised=False)[0]
                    #     save_single_image(result, sub_sample_path, f'result_shape_{scale}_rad_{rad_name}', to_normal=to_normal)

                    # for subject_j in radiomics_df['subject']:
                    #     x_rad = torch.tensor(radiomics_df[radiomics_df['subject'] == subject_j].values[0][2:].astype(np.float32)).unsqueeze(0).unsqueeze(2).to(self.config.training.device[0])
                    #     sample = net.sample(x_masking, context=x_rad, clip_denoised=False)[0]
                    #     save_single_image(result, sub_sample_path, f'result_shape_{scale}_rad_{subject_j}', to_normal=to_normal)

            else:
                if self.config.model.inference_type == "texture":
                    x_seg = test_batch['x_seg'].to(self.config.training.device[0])
                    x_crop, x_cond_crop, x_seg_crop, pos = test_batch['x_crop'], test_batch['x_cond_crop'], test_batch['x_seg_crop'], test_batch['pos']
                    x_next_seg = test_batch['x_next_seg'].to(self.config.training.device[0])
                    
                    x_crop = x_crop.to(self.config.training.device[0])
                    x_cond_crop = x_cond_crop.to(self.config.training.device[0])
                    # x_seg_crop = x_seg_crop.to(self.config.training.device[0])

                    x_next_rad = test_batch['x_next_rad'].unsqueeze(2).to(self.config.training.device[0])
                    sx, sy, sz, ex, ey, ez = pos["sx"].item(), pos["sy"].item(), pos["sz"].item(), pos["ex"].item(), pos["ey"].item(), pos["ez"].item()

                    x_seg_crop = x_seg[0,:,sx:ex, sy:ey, sz:ez]
                    x_cond_crop = x_rec[0,:,sx:ex, sy:ey, sz:ez].detach().clone()
                    # x_cond_crop[x_seg_crop == 1] = 0.5
                    # x_cond_crop[x_seg_crop == 4] = 1
                    x_cond_crop[x_seg_crop == 1] = 0.5
                    x_cond_crop[x_seg_crop == 2] = 1

                    x_remove_crop = net.sample(x_cond_crop, context=torch.tensor([[0.]]).to(self.config.training.device[0]), clip_denoised=False)[0]
                    x_remove = x_rec[0].detach().clone()
                    
                    va = 1
                    x_remove[:,sx+va:ex-va, sy+va:ey-va, sz+va:ez-va] = x_remove_crop[:,va:-va, va:-va, va:-va]

                    #x_next_seg[0,:,sx:ex, sy:ey, sz:ez]
                    x_ns_crop = test_batch['x_seg_crop'].to(self.config.training.device[0])[0]
                    x_cond_crop = x_remove[:,sx:ex, sy:ey, sz:ez].detach().clone()
                    # x_cond_crop[x_ns_crop == 1] = 0.5
                    # x_cond_crop[x_ns_crop == 4] = 1
                    x_cond_crop[x_ns_crop == 1] = 0.5
                    x_cond_crop[x_ns_crop == 2] = 1
                    x_cond_crop = x_cond_crop.to(self.config.training.device[0])

                    x_result_crop = net.sample(x_cond_crop, context=x_next_rad, clip_denoised=False)[0]
                    x_result = x_remove.detach().clone()
                    va = 1
                    x_result[:,sx+va:ex-va, sy+va:ey-va, sz+va:ez-va] = x_result_crop[:,va:-va, va:-va, va:-va]

                    x_seg = torch.zeros_like(x_next_seg[0])
                    x_seg[0,sx:ex, sy:ey, sz:ez] = x_ns_crop

                    subject_id = test_batch['subject_id'][0]
                    sub_sample_path = make_dir(os.path.join(sample_path, 'synth_samples',f'{subject_id}'))
                    save_single_image(x_seg, sub_sample_path, f'x_seg', to_normal=to_normal)
                    save_single_image(x_rec[0], sub_sample_path, f'recon', to_normal=to_normal)
                    save_single_image(x_remove, sub_sample_path, f'remove', to_normal=to_normal)
                    save_single_image(x_result, sub_sample_path, f'result', to_normal=to_normal)


                    
                # if self.config.model.inference_type == "texture":
                #     x_seg = test_batch['x_seg'].to(self.config.training.device[0])
                #     # x_remove = net.sample(x_cond[0:1], context=torch.tensor([[0.]]).to(self.config.training.device[0]), clip_denoised=False)

                #     radiomics_df = pd.read_csv(self.config.data.dataset_config.radiomics_path)
                #     max_rad = np.array(radiomics_df.max().iloc[2:]).astype(np.float32)
                #     min_rad = np.array(radiomics_df.min().iloc[2:]).astype(np.float32)
                #     max_rad = torch.tensor(max_rad).unsqueeze(0).unsqueeze(2).to(self.config.training.device[0])
                #     min_rad = torch.tensor(min_rad).unsqueeze(0).unsqueeze(2).to(self.config.training.device[0])
                #     zero_rad = torch.zeros_like(max_rad).to(self.config.training.device[0])
                #     min3_rad = -3 * torch.ones_like(max_rad).to(self.config.training.device[0])
                #     max3_rad = 3 * torch.ones_like(max_rad).to(self.config.training.device[0])
                #     rad_names = ["max", "min", "zero", "min3", "max3"]
                #     rads = [max_rad, min_rad, zero_rad, min3_rad, max3_rad]

                #     subject_id = test_batch['subject_id'][0]
                #     sub_sample_path = make_dir(os.path.join(sample_path, 'tumor_synthesis_samples',f'{subject_id}'))
                #     # save_single_image(x_remove[0], sub_sample_path, f'remove', to_normal=to_normal)
                #     save_single_image(x_seg[0], sub_sample_path, f'x_seg', to_normal=to_normal)
                #     # for i in range(batch_size):
                #     i = 0
                #     i_subject_id = test_batch['subject_id'][i]
                #     sub_sample_path = make_dir(os.path.join(sample_path, 'tumor_synthesis_samples',f'{i_subject_id}'))
                #     condition = x_cond[i].detach().clone()
                #     gt = x[i]
                #     recon = x_rec[i]
                #     save_single_image(condition, sub_sample_path, f'condition', to_normal=to_normal)
                #     save_single_image(gt, sub_sample_path, f'groundtruth', to_normal=to_normal)
                #     save_single_image(recon, sub_sample_path, f'recon', to_normal=to_normal)

                #     # x_masking = torch.zeros_like(x_seg)
                #     # x_masking[0] = get_masking3(x_rec,  x_seg[0], x_seg[0])
                #     # x_masking[1] = get_masking3(x_remove,  x_seg[0], x_seg[1])
                #     # x_masking[2] = get_masking3(x_remove,  x_seg[0], x_seg[2])
                #     # x_masking[3] = get_masking3(x_remove,  x_seg[0], x_seg[3])

                #     # x_masking = x_masking.to(self.config.training.device[0])

                #     for i in range(batch_size):
                #         i_subject_id = test_batch['subject_id'][i]
                #         result = net.sample(x_cond[i:i+1], context=x_rad[i:i+1], clip_denoised=False)[0]
                #         save_single_image(result, sub_sample_path, f'result_shape_{i_subject_id}', to_normal=to_normal)
                        
                #     for rad_name, rad in zip(rad_names, rads):
                #         result = net.sample(x_cond, context=rad, clip_denoised=False)[0]
                #         save_single_image(result, sub_sample_path, f'result_shape_{i_subject_id}_rad_{rad_name}', to_normal=to_normal)

                            # for j in range(batch_size):
                            #     j_subject_id = test_batch['subject_id'][j]
                            #     sample = net.sample(x_masking[i:i+1], context=x_rad[j:j+1], clip_denoised=False)
                            #     result = sample[0]
                            #     save_single_image(result, sub_sample_path, f'result_shape_{i_subject_id}_rad_{j_subject_id}', to_normal=to_normal)
                        # result = net.sample(x_cond[i:i+1], context=x_rad, clip_denoised=False)[0]
                        # save_single_image(result, sub_sample_path, f'result_rad', to_normal=to_normal)
                        # for rad_name, rad in zip(rad_names, rads):
                        #     result = net.sample(x_cond[i:i+1], context=rad, clip_denoised=False)[0]
                        #     save_single_image(result, sub_sample_path, f'result_shape_{i_subject_id}_rad_{rad_name}', to_normal=to_normal)
                        # for subject_j in radiomics_df['subject']:
                        #     x_rad = torch.tensor(radiomics_df[radiomics_df['subject'] == subject_j].values[0][2:].astype(np.float32)).unsqueeze(0).unsqueeze(2).to(self.config.training.device[0])
                        #     sample = net.sample(x_cond[i:i+1], context=x_rad, clip_denoised=False)
                        #     result = sample[0]
                        #     if subject_j == i_subject_id: save_single_image(result, sub_sample_path, f'result', to_normal=to_normal)
                        #     else: save_single_image(result, sub_sample_path, f'result_rad_{subject_j}', to_normal=to_normal)
                        # for j in range(batch_size):
                        #     j_subject_id = test_batch['subject_id'][j]
                        #     sample = net.sample(x_cond[i:i+1], context=x_rad[j:j+1], clip_denoised=False)
                        #     result = sample[0]
                        #     if i !=j: save_single_image(result, sub_sample_path, f'result_rad_{j_subject_id}', to_normal=to_normal)
                        #     else: save_single_image(result, sub_sample_path, f'result', to_normal=to_normal)

                else:
                    sub_sample_path = make_dir(os.path.join(sample_path, 'tumor_remove',f'{subject_id}'))
                    # x_seg = test_batch['x_seg']
                    # save_single_image(x_seg[0], sub_sample_path, f'x_seg', to_normal=to_normal)
                    for j in range(sample_num):
                        recon = x_rec[0]
                        save_single_image(recon, sub_sample_path, f'recon-2', to_normal=to_normal)
                        sample = net.sample(x_cond, context=torch.tensor([[0.]]).to(self.config.training.device[0]), clip_denoised=False)
                        # sample = net.sample_vqgan(x)
                        condition = x_cond[0].detach().clone()
                        gt = x[0]
                        result = sample[0]
                        recon = x_rec[0]
                        if j == 0:
                            save_single_image(condition, sub_sample_path, f'condition', to_normal=to_normal)
                            save_single_image(gt, sub_sample_path, f'groundtruth', to_normal=to_normal)
                            save_single_image(recon, sub_sample_path, f'recon', to_normal=to_normal)
                        if sample_num > 1:
                            save_single_image(result, sub_sample_path, f'result_{j}', to_normal=to_normal)
                        else:
                            save_single_image(result, sub_sample_path, f'result', to_normal=to_normal)


def flip_resize_rotate_around_center_3d(arr, scale_factors, rotation_angles):
    roi_coords = np.where(arr != 0)
    z_min, z_max = np.min(roi_coords[0]), np.max(roi_coords[0])
    y_min, y_max = np.min(roi_coords[1]), np.max(roi_coords[1])
    x_min, x_max = np.min(roi_coords[2]), np.max(roi_coords[2])
    
    roi = arr[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    
    flipped_roi = np.flip(np.flip(np.flip(roi, axis=0), axis=1), axis=2)
    
    resized_flipped_roi = zoom(flipped_roi, scale_factors, order=0)
    
    rotated_resized_flipped_roi = resized_flipped_roi
    for axis, angle in enumerate(rotation_angles):
        rotated_resized_flipped_roi = rotate(rotated_resized_flipped_roi, angle, axes=(axis, (axis + 1) % 3), reshape=False, order=0)
    
    center_z, center_y, center_x = map(int, measurements.center_of_mass(arr))
    start_z, start_y, start_x = (
        center_z - rotated_resized_flipped_roi.shape[0] // 2, 
        center_y - rotated_resized_flipped_roi.shape[1] // 2, 
        center_x - rotated_resized_flipped_roi.shape[2] // 2
    )
    end_z, end_y, end_x = (
        start_z + rotated_resized_flipped_roi.shape[0], 
        start_y + rotated_resized_flipped_roi.shape[1], 
        start_x + rotated_resized_flipped_roi.shape[2]
    )
    
    flipped_resized_rotated_arr = np.zeros_like(arr)
    flipped_resized_rotated_arr[start_z:end_z, start_y:end_y, start_x:end_x] = rotated_resized_flipped_roi
    
    return flipped_resized_rotated_arr

def get_masking(x_remove, x_seg, sacling, rotate):
    x_remove = x_remove.detach().cpu().numpy()[0,0]
    x_seg = x_seg.detach().cpu().numpy()[0,0]
    try: x_seg = flip_resize_rotate_around_center_3d(x_seg, (sacling, sacling, sacling), rotate)
    except: return None
    
    x_remove = np.array(x_remove)
    x_seg = np.array(x_seg)
    non_zero_indices = x_seg != 0
    x_remove[non_zero_indices] = x_seg[non_zero_indices]
    
    x_remove = torch.tensor(x_remove).unsqueeze(0).unsqueeze(0).to('cuda')
    return x_remove


def get_masking2(x_remove, x_seg, sacling, rotate):
    x_remove = x_remove.detach().cpu().numpy()[0,0]
    x_seg = x_seg.detach().cpu().numpy()[0,0]
    try: x_seg = flip_resize_rotate_around_center_3d(x_seg, (sacling, sacling, sacling), rotate)
    except: return None
    try: x_seg_2 = flip_resize_rotate_around_center_3d(x_seg, (sacling-0.1, sacling-0.1, sacling-0.1), rotate)
    except: return None
    
    x_remove = np.array(x_remove)
    x_seg = np.array(x_seg)
    x_seg_2 = np.array(x_seg_2)
    # non_zero_indices = x_seg != 0
    x_remove[x_seg_2 == 2] = 0.5
    x_remove[x_seg_2 == 1] = 0.5
    x_remove[x_seg == 2] = 1
    x_remove[x_seg == 1] = 1
    
    x_remove = torch.tensor(x_remove).unsqueeze(0).unsqueeze(0).to('cuda')
    return x_remove

def get_masking3(x_remove, src_seg, tag_seg):
    x_remove = x_remove.clone().detach().cpu().numpy()[0,0]
    src_seg = src_seg.clone().detach().cpu().numpy()[0]
    tag_seg = tag_seg.clone().detach().cpu().numpy()[0]

    cx, cy, cz = measurements.center_of_mass(src_seg)
    sx, sy, sz = measurements.center_of_mass(tag_seg)

    # Center shift
    dx, dy, dz = int(cx - sx), int(cy - sy), int(cz - sz)
    tag_seg = np.roll(tag_seg, (dx, dy, dz), axis=(0, 1, 2))
    
    x_remove = np.array(x_remove)
    x_seg = np.array(tag_seg)
    x_seg[x_seg == 2] = 0
    x_seg[x_seg == 1] = 0.5
    x_seg[x_seg == 4] = 1
    non_zero_indices = x_seg != 0
    x_remove[non_zero_indices] = x_seg[non_zero_indices]
    
    x_remove = torch.tensor(x_remove).unsqueeze(0).unsqueeze(0).to('cuda')
    return x_remove