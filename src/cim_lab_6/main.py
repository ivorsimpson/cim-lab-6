import json
import time
from pathlib import Path

import deepinv as dinv
import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image

from .data import load_image
from .dps import load_denoiser, sample
from .metrics import PSNR, evaluate
from .operators import make_physics


def device_from(name):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def noise_std(noise_cfg):
    """Measurement noise std (for Poisson, at a mid-grey pixel value of 0.5)."""
    return noise_cfg.sigma if noise_cfg.model == "gaussian" else (0.5 * noise_cfg.gain) ** 0.5


def show(images, titles, path):
    fig, axes = plt.subplots(1, len(images), figsize=(2.2 * len(images), 2.5))
    for ax, im, title in zip(axes, images, titles):
        a = im[0].detach().cpu()
        if title.startswith(("std", "|error|")):
            ax.imshow(a.mean(0), cmap="magma")
        else:
            ax.imshow(a.permute(1, 2, 0).clamp(0, 1))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reconstruct_all(cfg, names, denoiser, device, out=None):
    """Simulate a measurement of each named image, run DPS, and score the result."""
    physics = make_physics(cfg.operator, cfg.data.image_size, device)
    sigma_y = noise_std(cfg.operator.noise)
    rows = []
    for k, name in enumerate(names):
        x = load_image(name, cfg.data.root, cfg.data.image_size).to(device)
        torch.manual_seed(cfg.seed + k)
        y = physics(x)
        baseline = physics.A_dagger(y).clamp(0, 1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        samples = sample(y, physics, denoiser, cfg.sampler, cfg.data.image_size, sigma_y, cfg.seed + k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        row = {"image": name, "seconds": seconds, "psnr_baseline": PSNR(baseline, x).item()}
        row |= evaluate(samples, x, y, physics, sigma_y)
        rows.append(row)
        print(", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        if out is not None:
            stem = name.replace("dinv:", "").replace("/", "_").rsplit(".", 1)[0]
            mean = samples.mean(0, keepdim=True)
            std = samples.std(0, keepdim=True) if len(samples) > 1 else torch.zeros_like(mean)
            n = min(len(samples), 3)
            show(
                [y, x, baseline, *samples[:n, None], mean, std, (mean - x).abs()],
                ["measurements", "target", "pseudo-inverse A†y", *[f"sample {i}" for i in range(n)],
                 "posterior mean", "std (uncertainty)", "|error| of mean"],
                out / f"{stem}.png",
            )
            save_image(samples, out / f"{stem}_samples.png", nrow=len(samples))
    return rows


@hydra.main(version_base=None, config_path="../../conf", config_name="experiment")
def main(cfg: DictConfig):
    device = device_from(cfg.device)
    out = Path.cwd()
    print(OmegaConf.to_yaml(cfg))
    print(f"Using {device}; outputs: {out}")
    denoiser = load_denoiser(cfg.model.name, device)

    if cfg.mode == "generate":
        # Unconditional samples from the prior: what does the model think images look like?
        torch.manual_seed(cfg.seed)
        cfg.sampler.guidance = "none"
        size = cfg.data.image_size
        dummy = torch.zeros(1, 3, size, size, device=device)
        faces = sample(dummy, dinv.physics.Denoising(), denoiser, cfg.sampler, size, 1.0, cfg.seed)
        for i, face in enumerate(faces):
            save_image(face, out / f"prior_{i:02d}.png")
        print(f"Saved {len(faces)} prior samples to {out}")
        return

    if cfg.mode != "reconstruct":
        raise ValueError(f"Unknown mode: {cfg.mode}")
    rows = reconstruct_all(cfg, list(cfg.data.names), denoiser, device, out)
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "image"}
    print("Average:", {k: round(v, 3) for k, v in summary.items()})
    (out / "metrics.json").write_text(
        json.dumps({"config": OmegaConf.to_container(cfg), "mean": summary, "per_image": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
