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
        apply_gradient_penalty_every = 4,
        calc_radiomics_loss_every = 0,
        discr_aux_recon_loss_weight = 1,
        learning_rate = 5e-6,
        generator = dict(
            dim_capacity = 8,
            image_size = 32,
            dim_max = 512,
            num_skip_layers_excite = 4,
            unconditional = False,
            cross_attn_resolutions=(16,8,4),
            self_attn_resolutions=(8,),
            channels = 2,
            dim_latent = 512,
            self_attn_dim_head = 32,
            self_attn_heads = 16,
            self_attn_ff_mult = 16,
            cross_attn_dim_head = 32,
            cross_attn_heads = 16,
            cross_attn_ff_mult = 16,
            num_conv_kernels = 2,  # the number of adaptive conv kernels
            pixel_shuffle_upsample = False, # False, # False
            latent_size = (4,4,4),
        ),
        discriminator = dict(
            dim_capacity = 8,
            dim_max = 512,
            image_size = 32,
            num_skip_layers_excite = 4,
            unconditional = False,
            channels = 2,
            attn_dim_head = 32,
            attn_heads = 16,
            text_dim = 34, # 51
            latent_size = (4,4,4),
            aux_recon_resolutions = (8,),
            aux_recon_patch_dims=(2,),
        ),
        amp = False,
        model_folder = './gigagan-models/brats',
        results_folder = './gigagan-results/brats',
    ).cuda()

    # dataset
    dataset = BraTSDataset(
        dataset_path = 'BraTS2021/TrainingData',
        radiomics_path = 'radiomics/shape_features.csv',
        img_size = 128,
        resize = (32,32,32)
    )

    dataloader = dataset.get_dataloader(batch_size = 1, num_workers=8)

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