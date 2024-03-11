import os
import numpy as np
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.nn import L1Loss
from tqdm import tqdm

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from generative.networks.nets import VQVAE

from dataset import get_brats_dataset, Sampler

print_config()

parser = argparse.ArgumentParser(description="VQVAE training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="vqvae", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--data_path", default='BraTS2021/TrainingData', type=str, help="dataset directory")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=2000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--embedding_dim", default=64, type=int, help="feature size")
parser.add_argument("--num_embeddings", default=8192, type=int, help="number of embeddings")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--perceptual_weight", default=0.001, type=float, help="perceptual loss weight")
parser.add_argument("--adv_weight", default=0.001, type=float, help="adversarial loss weight")
parser.add_argument("--roi_weight", default=10.0, type=float, help="ROI loss weight")
parser.add_argument("--num_channels", default=256, type=int, help="number of channels")


def save_checkpoint(autoencoder, discriminator, epoch, args, filename="model.pt", best_loss=0, optimizer=None, optimizer_d=None, scheduler=None):
    state_dict = autoencoder.state_dict() if not args.distributed else autoencoder.module.state_dict()
    discriminator_dict = discriminator.state_dict() if not args.distributed else discriminator.module.state_dict()
    save_dict = {"epoch": epoch, "best_loss": best_loss, "state_dict": state_dict, "discriminator": discriminator_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if optimizer_d is not None:
        save_dict["optimizer_d"] = optimizer_d.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    maybe_mkdir(args.logdir)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def maybe_mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass

def get_random_int(min, max):
    randint = np.random.randint(min, max, size=2)
    while randint[0] == randint[1]:
        randint = np.random.randint(min, max, size=2)
    return randint

def train_epoch(autoencoder, discriminator, optimizer, optimizer_d, l1_loss, loss_perceptual, adv_loss, progress_bar, args, device):
    epoch_loss = 0
    quant_loss = 0
    gen_epoch_loss = 0
    roi_epoch_loss = 0
    disc_epoch_loss = 0

    for step, batch in progress_bar:
        mov_image = batch['image'].to(device)
        seg_image = batch['seg'].to(device)
        
        # Generator part
        optimizer.zero_grad(set_to_none=True)
        reconstruction, quantization_loss = autoencoder(images=mov_image)
        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
        # reconstruction loss
        recons_loss = l1_loss(reconstruction.float(), mov_image.float())
        # perceptual loss
        p_loss = loss_perceptual(reconstruction.float(), mov_image.float())
        # adversarial loss
        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        # ROI loss
        seg_image[seg_image!=0] = 1
        roi_loss = l1_loss(reconstruction.float() * seg_image.float(), mov_image.float() * seg_image.float())

        loss = recons_loss + quantization_loss + args.perceptual_weight * p_loss + args.adv_weight * generator_loss + args.roi_weight * roi_loss
        loss.backward()
        optimizer.step()

        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)
        
        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(mov_image.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = args.adv_weight * discriminator_loss
        loss_d.backward()
        optimizer_d.step()

        epoch_loss += recons_loss.item()
        quant_loss += quantization_loss.item()
        gen_epoch_loss += generator_loss.item()
        roi_epoch_loss += roi_loss.item()
        disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recon": epoch_loss / (step + 1),
                "quant": quant_loss/ (step + 1),
                "gen": gen_epoch_loss / (step + 1),
                "roi": roi_epoch_loss / (step + 1),
                "disc": disc_epoch_loss / (step + 1),
            }
        )

    return progress_bar

def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    set_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = get_brats_dataset(args.data_path)

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              num_workers=8,
                              pin_memory=True)

    autoencoder = VQVAE(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            num_channels=[args.num_channels],
            num_res_channels=args.num_channels,
            num_res_layers=2,
            downsample_parameters=[(2, 4, 1, 1)],
            upsample_parameters=[(2, 4, 1, 1, 0)],
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
        )
    autoencoder.to(device)

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)

    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
    discriminator.to(device)

    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_autoencoder_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            if k in autoencoder.state_dict():
                if v.shape != autoencoder.state_dict()[k].shape:
                    print("skipping", k)
                    continue
                new_autoencoder_state_dict[k.replace("backbone.", "")] = v
            else:
                print("skipping", k)

        autoencoder.load_state_dict(new_autoencoder_state_dict, strict=False)

        new_discriminator_state_dict = OrderedDict()
        for k, v in checkpoint["discriminator"].items():
            new_discriminator_state_dict[k.replace("backbone.", "")] = v
        discriminator.load_state_dict(new_discriminator_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, start_epoch))

    autoencoder.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        autoencoder.cuda(args.gpu)
        autoencoder = torch.nn.parallel.DistributedDataParallel(autoencoder, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True, broadcast_buffers=False)
        discriminator.cuda(args.gpu)
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True, broadcast_buffers=False)
        
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=args.optim_lr)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.optim_lr)

    save_interval = 5

    autoencoder.train()
    discriminator.train()
    for epoch in range(start_epoch, args.max_epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=160)
        progress_bar.set_description(f"Epoch {epoch}")
        progress_bar = train_epoch(autoencoder, discriminator, optimizer, optimizer_d, l1_loss, loss_perceptual, adv_loss, progress_bar, args, device)
        
        if epoch % save_interval == 0:
            autoencoder.eval()
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(autoencoder, discriminator, epoch, args, optimizer=optimizer, optimizer_d=optimizer_d, filename=f"model_ep{epoch:04}.pt")


if __name__ == "__main__":
    main()