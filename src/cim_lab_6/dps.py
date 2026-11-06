"""Diffusion posterior sampling (Chung et al., 2023, https://arxiv.org/abs/2209.14687).

The loop below is Algorithm 1 of the DPS paper written with DDIM-style steps, so that
the number of steps can be reduced and `eta` can switch between stochastic (eta=1,
DDPM-like) and deterministic (eta=0, DDIM) sampling.

Each step:
  1. Tweedie: estimate the clean image x0_hat = E[x0 | x_t] with the pretrained denoiser.
  2. Prior step: move x_t to x_{t-1} using x0_hat (ordinary unconditional diffusion).
  3. Guidance: nudge x_{t-1} down the gradient of a data-fidelity loss,
     loss(y - A(x0_hat)), differentiated through the denoiser with respect to x_t.
"""

import deepinv as dinv
import torch

# The DDPM noise schedule used to train the FFHQ / ImageNet models (Ho et al., 2020).
BETAS = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
ALPHA_BAR = torch.cumprod(1 - BETAS, dim=0)


def load_denoiser(name: str, device) -> dinv.models.Denoiser:
    """Pretrained diffusion denoisers D(x, sigma) for images in [0, 1]."""
    if name == "ffhq256":
        model = dinv.models.DiffUNet(pretrained="download")
    elif name == "imagenet256":
        model = dinv.models.DiffUNet(large_model=True, pretrained="download")
    elif name == "ffhq64":
        model = dinv.models.NCSNpp(pretrained="download")
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device).eval().requires_grad_(False)


def predict_x0(denoiser, x_t, alpha_bar):
    """
    Calculate the Tweedie estimate of the clean image, both in [-1, 1].
    Using DDPM notation x_t = sqrt(a) x0 + sqrt(1 - a) eps, with x0 in [-1, 1]. DeepInverse
    denoisers expect z = x0' + sigma * noise with x0' in [0, 1], so we rescale.
    """
    z = (x_t / alpha_bar.sqrt() + 1) / 2
    sigma = ((1 - alpha_bar) / alpha_bar).sqrt() / 2
    return 2 * denoiser(z, sigma.float()) - 1


def guidance_loss(residual, cfg, sigma_y, sigma_t):
    """Data-fidelity loss whose gradient guides the sampler.

    residual: y - A(x0_hat), one row per sample in the batch.
    sigma_y:  measurement noise standard deviation.
    sigma_t:  noise level of x_t, i.e. the std of x_t / sqrt(alpha_bar) around x0.
    All three are on the [-1, 1] image scale used by the DPS paper.
    Losses are summed over the batch: each sample only depends on its own x_t.
    """
    sq_norm = residual.flatten(1).square().sum(1)
    if cfg.guidance == "dps":  # the DPS paper: zeta * grad ||y - A x0_hat||
        return sq_norm.sqrt().sum()
    if cfg.guidance == "squared":  # the "obvious" Gaussian log-likelihood
        return 0.5 * sq_norm.sum()
    if cfg.guidance == "annealed":  # likelihood with variance sigma_y^2 + sigma_t^2
        return (sq_norm / (2 * (sigma_y**2 + sigma_t**2))).sum()
    if cfg.guidance == "none":  # unconditional: ignore the measurement
        return 0.0 * sq_norm.sum()
    if cfg.guidance == "custom":
        raise NotImplementedError("Task: design your own guidance term")
    raise ValueError(f"Unknown guidance: {cfg.guidance}")


def dps(y, physics, denoiser, cfg, shape, sigma_y, generator=None):
    """Draw len(y) posterior samples. Returns images in [0, 1]."""
    device = y.device
    alpha_bar = ALPHA_BAR.float().to(device)
    timesteps = torch.linspace(999, 0, cfg.num_steps).round().long().tolist()
    x = torch.randn(shape, device=device, generator=generator)

    for i, t in enumerate(timesteps):
        a = alpha_bar[t]
        a_prev = alpha_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.ones_like(a)

        x = x.detach().requires_grad_(cfg.guidance != "none")
        with torch.set_grad_enabled(cfg.guidance != "none"):
            x0 = predict_x0(denoiser, x, a).clamp(-1, 1)
            # x2 puts the residual on the [-1, 1] scale used in the DPS paper, so that
            # `weight` matches the paper's zeta (they use zeta = 1 for super-resolution).
            residual = 2 * (y - physics.A((x0 + 1) / 2))

        # Prior (DDIM) step: re-noise x0_hat to the next noise level.
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
        c = cfg.eta * ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).sqrt()
        noise = torch.randn(shape, device=device, generator=generator)
        x_next = a_prev.sqrt() * x0 + (1 - a_prev - c**2).sqrt() * eps + c * noise

        # Likelihood (guidance) step.
        if cfg.guidance != "none":
            sigma_t = ((1 - a) / a).sqrt()
            loss = guidance_loss(residual, cfg, 2 * sigma_y, sigma_t)
            grad = torch.autograd.grad(loss, x)[0]
            x_next = x_next - cfg.weight * grad
        x = x_next.detach().float()

    return ((x0.detach() + 1) / 2).clamp(0, 1)


def sample(y, physics, denoiser, cfg, image_size, sigma_y, seed=0):
    """Posterior samples for one measurement y (batch of 1) -> (n_samples, 3, H, W)."""
    shape = (cfg.n_samples, 3, image_size, image_size)
    ys = y.expand(cfg.n_samples, *y.shape[1:])
    if cfg.name == "ours":
        generator = torch.Generator(y.device).manual_seed(seed)
        return dps(ys, physics, denoiser, cfg, shape, sigma_y, generator)
    if cfg.name == "deepinv":
        # DeepInverse's own implementation: the same algorithm with its own conventions,
        # so `weight` is on a different scale; guidance must be "dps" or "annealed".
        sampler = dinv.sampling.DPS(
            denoiser, num_steps=cfg.num_steps, alpha=cfg.eta, weight=cfg.weight,
            guidance="norm" if cfg.guidance == "dps" else cfg.guidance,
            device=y.device, dtype=torch.float32,
        )
        return sampler(ys, physics, x_init=shape, seed=seed).clamp(0, 1)
    raise ValueError(f"Unknown sampler: {cfg.name}")
