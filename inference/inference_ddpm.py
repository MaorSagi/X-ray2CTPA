import logging
import os
import torch
import numpy as np
import SimpleITK as sitk
from omegaconf import OmegaConf
from ddpm import Unet3D, GaussianDiffusion  # <-- Make sure your PYTHONPATH includes project root

log = logging.getLogger(__name__)
log.info("Inference started")

# === Load config ===
cfg = OmegaConf.load("config/base_cfg.yaml")  # Adjust path if needed

# === GPU Setup ===
device = torch.device(f"cuda:{cfg.model.gpus}" if torch.cuda.is_available() else "cpu")

# === Initialize UNet ===
model = Unet3D(
    dim=cfg.model.diffusion_img_size,
    cond_dim=cfg.model.cond_dim,
    dim_mults=cfg.model.dim_mults,
    channels=cfg.model.diffusion_num_channels,
    resnet_groups=8,
    classifier_free_guidance=cfg.model.classifier_free_guidance,
    medclip=cfg.model.medclip
).to(device)

# === Initialize GaussianDiffusion ===
diffusion = GaussianDiffusion(
    model,
    vqgan_ckpt=cfg.model.vqgan_ckpt,
    vae_ckpt=cfg.model.vae_ckpt,
    image_size=cfg.model.diffusion_img_size,
    num_frames=cfg.model.diffusion_depth_size,
    channels=cfg.model.diffusion_num_channels,
    timesteps=cfg.model.timesteps,
    img_cond='xray' in cfg.model.name_dataset.lower(),
    loss_type=cfg.model.loss_type,
    l1_weight=cfg.model.l1_weight,
    perceptual_weight=cfg.model.perceptual_weight,
    discriminator_weight=cfg.model.discriminator_weight,
    classification_weight=cfg.model.classification_weight,
    classifier_free_guidance=cfg.model.classifier_free_guidance,
    medclip=cfg.model.medclip,
    name_dataset=cfg.model.name_dataset,
    dataset_min_value=cfg.model.dataset_min_value,
    dataset_max_value=cfg.model.dataset_max_value,
).to(device)

# === Load model checkpoint ===
milestone = cfg.model.load_milestone
ckpt_path = os.path.join(cfg.model.results_folder, f"model-{milestone}.pt")
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"], strict=False)

# === Load or fake CXR ===
# cxr_path = "path/to/xray_input.npy"
# cxr = torch.from_numpy(np.load(cxr_path)).float().unsqueeze(0).to(device)  # [1, 1, H, W]
# ecg = torch.
# === Generate sample ===
# diffusion.eval()
# with torch.no_grad():
#     sample = diffusion.sample(cond=cxr, batch_size=1, cond_scale=1.0)  # shape: [1, 1, D, H, W]

# === Save output ===
# output_volume = sample[0, 0].cpu().numpy()
# sitk_image = sitk.GetImageFromArray(output_volume)
# sitk.WriteImage(sitk_image, "generated_ctpa.mha")  # or use np.save("ctpa.npy", output_volume)

# === Predict ===
diffusion.eval()
with torch.no_grad():
    pred = diffusion.predict(cond=, batch_size=1, cond_scale=1.0)  # shape: [1, 1, D, H, W]
