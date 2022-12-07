"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import math
import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision as tv
from PIL import Image

from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.clip import (
    ModifiedResNet,
    VisionTransformer
)
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    adjust_scales2image
)
from guided_diffusion.utils import (
    compute_cosine_distance,
    leftupper_coords_from_size
)


def main():
    args = create_argparser().parse_args()
    args.clip_layers = [2]

    dist_util.setup_dist()
    logger.configure()

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)

    real = tv.transforms.ToTensor()(Image.open(args.data_dir))[None]
    adjust_scales2image(real, args)
    models = []
    diffusions = []
    for current_scale in range(args.stop_scale + 1)[-1:]:
        args.class_cond = False # if current_scale == 0 else True
        # args.num_channels = min(args.num_channels_init * pow(2, math.floor(current_scale / 2)), 512)
        # args.num_res_blocks = min(args.num_res_blocks_init + math.floor(current_scale / 4), 6)
        args.model_path = os.path.join(args.model_root, 'scale_8', 'ema_0.9999_100000.pt')
        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        model.to(dist_util.dev())

        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        models.append(model)
        diffusions.append(diffusion)

    logger.log("sampling...")
    for current_scale in range(args.stop_scale + 1)[-1:]:
        model, diffusion = models[0], diffusions[0]
        current_factor = math.pow(args.scale_factor, args.stop_scale - current_scale)
        curr_h, curr_w = round(args.full_size[0] * current_factor), round(args.full_size[1] * current_factor)
        curr_h_pad, curr_w_pad = math.ceil(curr_h / 8) * 8, math.ceil(curr_w / 8) * 8
        pad_size = (0, curr_w_pad - curr_w, 0, curr_h_pad - curr_h)

        model_kwargs = {}

        if any(pad_size):
            model_kwargs["pad_size"] = pad_size

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, curr_h, curr_w),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            progress=True
        )
        batch_prev = sample.clone()
        os.makedirs(args.results_path + "scale_" + str(current_scale), exist_ok=True)
        for i in range(sample.shape[0]):
            tv.utils.save_image(sample[i] * 0.5 + 0.5, args.results_path + "scale_" + str(current_scale) + '/%d.png' % (i+8))
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10000,
        full_size=(166, 500),
        num_channels_init=512,
        num_res_blocks_init=6,
        scale_factor_init=0.75,
        min_size=25,
        max_size=250,
        nc_im=3,
        batch_size=8,
        use_ddim=False,
        model_path="",
        model_root="",
        classifier_scale=10000.0,
        results_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
