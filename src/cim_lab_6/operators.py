"""Forward models y = A(x) + noise, all built from DeepInverse physics."""

import deepinv as dinv
import torch


def make_noise(cfg):
    if cfg.model == "gaussian":
        return dinv.physics.GaussianNoise(sigma=cfg.sigma)
    if cfg.model == "poisson":
        return dinv.physics.PoissonNoise(gain=cfg.gain, clip_positive=True)
    raise ValueError(f"Unknown noise model: {cfg.model}")


def make_physics(cfg, image_size: int, device) -> dinv.physics.Physics:
    shape = (3, image_size, image_size)
    noise = make_noise(cfg.noise)
    name = cfg.name

    if name == "super_resolution":
        return dinv.physics.Downsampling(
            img_size=shape, factor=cfg.sr_factor, filter="bicubic",
            padding="circular", noise_model=noise, device=device,
        )
    if name in {"gaussian_blur", "motion_blur"}:
        if name == "gaussian_blur":
            psf = dinv.physics.functional.gaussian_blur(sigma=(cfg.blur_sigma, cfg.blur_sigma))
        else:
            generator = dinv.physics.generator.MotionBlurGenerator(
                psf_size=(31, 31), l=cfg.motion_length,
                rng=torch.Generator().manual_seed(0),  # the same blur every run
            )
            psf = generator.step(batch_size=1)["filter"]
        return dinv.physics.Blur(
            filter=psf.to(device), padding="circular", noise_model=noise, device=device
        )
    if name == "inpainting_box":
        mask = torch.ones(1, image_size, image_size)
        side = int(cfg.box_fraction * image_size)
        start = (image_size - side) // 2
        mask[:, start : start + side, start : start + side] = 0
        return dinv.physics.Inpainting(
            img_size=shape, mask=mask, noise_model=noise, device=device
        )
    if name == "inpainting_random":
        return dinv.physics.Inpainting(
            img_size=shape, mask=cfg.keep_fraction, noise_model=noise, device=device,
            rng=torch.Generator(device).manual_seed(0),
        )
    if name == "colourise":
        return dinv.physics.Decolorize(noise_model=noise, device=device)
    raise ValueError(f"Unknown operator: {name}")
