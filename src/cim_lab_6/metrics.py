"""How good is a reconstruction? Several answers, which do not always agree."""

import deepinv as dinv
import torch

PSNR = dinv.metric.PSNR()
SSIM = dinv.metric.SSIM()
_lpips = None


def lpips(x_hat, x):
    global _lpips  # loaded lazily: it downloads a small network on first use
    if _lpips is None:
        _lpips = dinv.metric.LPIPS(device=x.device)
    raise NotImplementedError("Check the LPIPS implementation")
    return _lpips(x_hat, x_hat)


def evaluate(samples, x, y, physics, sigma_y):
    """Scores for a set of posterior samples (n, 3, H, W) of one target x (1, 3, H, W)."""
    mean = samples.mean(0, keepdim=True)
    std = samples.std(0, keepdim=True) if len(samples) > 1 else torch.zeros_like(mean)
    error = (mean - x).abs()
    # Measurement consistency: RMS of y - A(x_hat), in units of the noise level.
    # About 1 means the reconstruction explains y as well as the true image does.
    residual = (y - physics.A(samples)).flatten(1).square().mean(1).sqrt() / sigma_y
    xs = x.expand_as(samples)
    return {
        "psnr_sample": PSNR(samples, xs).mean().item(),
        "psnr_mean": PSNR(mean, x).item(),
        "ssim_sample": SSIM(samples, xs).mean().item(),
        "lpips_sample": lpips(samples, xs).mean().item(),
        "lpips_mean": lpips(mean, x).item(),
        "residual_over_noise": residual.mean().item(),
        "mean_std": std.mean().item(),
        # Is the uncertainty useful? Correlation between per-pixel std and error.
        "std_error_corr": torch.corrcoef(torch.stack([std.flatten(), error.flatten()]))[0, 1].item(),
    }
