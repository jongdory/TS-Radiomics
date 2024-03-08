import torch
import SimpleITK as sitk
from torch.cuda.amp import autocast

from gigagan_pytorch import (
    GigaGAN,
    BraTSDataset
)

gan = GigaGAN(
    generator = dict(
        dim_capacity = 8,
        style_network = dict(
            dim = 32,
            depth = 4,
        ),
        image_size = 64,
        dim_max = 256,
        num_skip_layers_excite = 4,
        unconditional = False,
        cross_attn_resolutions=(32,16,8)
    ),
    discriminator = dict(
        dim_capacity = 8,
        dim_max = 256,
        image_size = 64,
        num_skip_layers_excite = 4,
        unconditional = False
    ),
    amp = True
).cuda()

# dataset

dataset = BraTSDataset(
    dataset_path = '/store8/01.Database/01.Brain/04.BraTS2021/TrainingData',
    radiomics_path = '/store8/05.IntracranialArtery/radiomics_features/brats_shape_features.csv',
)

dataloader = dataset.get_dataloader(batch_size = 1)

# you must then set the dataloader for the GAN before training

gan.set_dataloader(dataloader)

gan.load('gigagan-models/model-12.ckpt')
# after much training

radiomics = dataset.imgs[0]['radiomics'][None, :]

images = gan.generate(texts=radiomics)

sitk.WriteImage(sitk.GetImageFromArray(images[0,0].cpu().numpy()), 'test.nii.gz')