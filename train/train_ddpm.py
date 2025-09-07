import time
start = time.time()
from re import I
import os
import torch
from ddpm import Unet3D, GaussianDiffusion, Trainer
# from dataset import MRNetDataset, BRATSDataset
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from train.get_dataset import get_dataset


from ddpm.unet import UNet
from datetime import date
import logging
import wandb
log = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

def timer(start,end, msg=''):
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"{msg}-"+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    global start
    torch.cuda.set_device(cfg.model.gpus)

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    wandb.init(project="ECGXray2CTPA", config=flatten_dict(cfg))


    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            cond_dim=cfg.model.cond_dim,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            resnet_groups=8,
            classifier_free_guidance = cfg.model.classifier_free_guidance,
            medclip = cfg.model.medclip
        ).cuda()
    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        vae_ckpt=cfg.model.vae_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        img_cond = 'xray' in cfg.model.name_dataset.lower(),
        img_cond_dim = cfg.model.img_cond_dim,
        ecg_cond='ecg' in cfg.model.name_dataset.lower(),
        ecg_cond_dim=cfg.model.ecg_cond_dim,
        loss_type=cfg.model.loss_type,
        l1_weight = cfg.model.l1_weight,
        perceptual_weight = cfg.model.perceptual_weight,
        discriminator_weight = cfg.model.discriminator_weight,
        classification_weight = cfg.model.classification_weight,
        classifier_free_guidance = cfg.model.classifier_free_guidance,
        medclip = cfg.model.medclip,
        name_dataset = cfg.model.name_dataset,
        dataset_min_value = cfg.model.dataset_min_value,
        dataset_max_value = cfg.model.dataset_max_value,
    ).cuda()

    train_dataset, val_dataset, _ = get_dataset(cfg)

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        max_grad_norm=cfg.model.max_grad_norm,
        lora = cfg.model.lora,
        lora_first = cfg.model.lora_first,
        warmup_steps=cfg.model.warmup_steps,
        hard_warmup=cfg.model.hard_warmup,
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone, map_location='cuda:'+str(cfg.model.gpus))
    timer(start, time.time(),msg="From starting program till training model")
    start = time.time()
    trainer.train()
    timer(start, time.time(),msg="From starting program till the end")


if __name__ == '__main__':
    run()

