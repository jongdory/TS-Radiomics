import numpy as np
import SimpleITK as sitk
import torch
from torch.cuda.amp import autocast

from gigagan_pytorch import (
    GigaGAN,
    BraTSDataset
)

def main():
    gan = GigaGAN(
        calc_multiscale_loss_every = 2,
        apply_gradient_penalty_every = 0,
        calc_radiomics_loss_every = 0,
        discr_aux_recon_loss_weight = 0.1,
        matching_awareness_loss_weight=1,
        learning_rate = 1e-6,
        train_upsampler = True,
        upsampler_replace_rgb_with_input_lowres_image = True,
        generator = dict(
            dim = 16,
            image_size = 128,
            input_image_size = 32,
            dim_mults = (1, 1, 2, 4, 8),
            text_dim = 34,
            channels = 2,
            resnet_block_groups = 8,
            full_attn = (False, False, False, False, False),
            cross_attn = (False, False, False, False, True),
            flash_attn = False,
            self_attn_dim_head = 32,
            self_attn_heads = 4,
            self_attn_dot_product = True,
            self_attn_ff_mult = 4,
            attn_depths = (1, 1, 1, 1, 1),
            cross_attn_dim_head = 32,
            cross_attn_heads = 4,
            cross_ff_mult = 4,
            mid_attn_depth = 1,
            num_conv_kernels = 2,
            resize_mode = 'trilinear',
            unconditional = False,
            skip_connect_scale = None,
            pixel_shuffle_upsample=False,
        ),
        discriminator = dict(
            multiscale_input_resolutions = (64,),
            dim_capacity = 8,
            dim_max = 256,
            image_size = 128,
            num_skip_layers_excite = 4,
            unconditional = False,
            channels = 2,
            attn_dim_head = 32,
            attn_heads = 8,
            text_dim = 34, 
            latent_size = (4,4,4),
            aux_recon_patch_dims=(2,),
        ),
        amp = False,
        model_folder = './gigagan-models/brats-up',
        results_folder = './gigagan-results/brats-up',
    ).cuda()

    # dataset
    dataset = BraTSDataset(
        dataset_path = 'BraTS2021/TrainingData',
        radiomics_path = 'radiomics/shape_features.csv',
        img_size = 128,
    )

    dataloader = dataset.get_dataloader(batch_size=1, num_workers=8)

    # you must then set the dataloader for the GAN before training
    gan.set_dataloader(dataloader)
    # gan.load('gigagan-models/brats/model.ckpt')
    # training the discriminator and generator alternating
    # for 100 steps in this example, batch size 1, gradient accumulated 8 times

    gan(
        steps = 500000,
        grad_accum_every = 8
        )

if __name__ == '__main__':
    main()