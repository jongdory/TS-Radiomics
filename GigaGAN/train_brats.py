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
        calc_multiscale_loss_every = 4,
        apply_gradient_penalty_every = 0,
        learning_rate = 5e-6,
        generator = dict(
            dim_capacity = 8,
            image_size = 64,
            dim_max = 256,
            num_skip_layers_excite = 4,
            unconditional = False,
            cross_attn_resolutions=(32, 16,8),
            self_attn_resolutions=(32,),
            channels = 3,
            dim_latent = 256,
            self_attn_dim_head = 32,
            self_attn_heads = 8,
            self_attn_ff_mult = 8,
            cross_attn_dim_head = 32,
            cross_attn_heads = 8,
            cross_attn_ff_mult = 8,
            num_conv_kernels = 2, 
            pixel_shuffle_upsample = False, 
            latent_size = (6,6,6)
        ),
        discriminator = dict(
            dim_capacity = 8,
            dim_max = 256,
            image_size = 64,
            num_skip_layers_excite = 4,
            unconditional = False,
            channels = 3,
            attn_dim_head = 32,
            attn_heads = 8,
            text_dim = 51,
            latent_size = (6,6,6),
        ),
        amp = False,
        model_folder = './gigagan-models/brats',
        results_folder = './gigagan-results/brats',
    ).cuda()

    # dataset
    dataset = BraTSDataset(
        dataset_path = '/store8/01.Database/01.Brain/04.BraTS2021/TrainingData',
        radiomics_path = '/store8/05.IntracranialArtery/radiomics_features/brats_shape_features.csv',
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