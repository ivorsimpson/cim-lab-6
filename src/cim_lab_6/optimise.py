"""Part G: hyperparameter search for the DPS sampler.

    uv run python -m cim_lab_6.optimise
    uv run python -m cim_lab_6.optimise operator.name=inpainting_box search.strategy=grid

Settings are in conf/optimise.yaml, which extends conf/experiment.yaml. The workflow is
the standard one for tuning any ML system, not just DPS:

  1. define a search space              TASK 1  (conf/optimise.yaml: space)
  2. generate candidate configurations  TASK 2  (sample_value, grid_values)
  3. score each on validation images    TASK 3  (objective)
  4. re-check the most promising ones   TASK 4  (confirm_top_k)
  5. report the winner on test images   provided: compare it with the default settings

Outputs (in the Hydra run folder): trials.csv (every configuration and its scores),
trials.png (objective against each hyperparameter) and report.yaml.
"""

import csv
import itertools
import math
import random
import statistics
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import yaml
from omegaconf import DictConfig, OmegaConf

from .dps import load_denoiser
from .main import device_from, reconstruct_all


def sample_value(spec, rng: random.Random):
    """TASK 2a: draw ONE random value for one hyperparameter (random search).

    `spec` is one entry of `space` in conf/optimise.yaml: either {values: [...]} or
    {low, high, scale}. For scale=log, sample uniformly in log(value): a value between
    0.01 and 0.1 should be as likely as one between 1 and 10. Use `rng` for all randomness
    (rng.choice, rng.uniform) so that runs are reproducible.
    """
    raise NotImplementedError("TASK 2a: sample one value from spec")


def grid_values(spec) -> list:
    """TASK 2b: ALL the values grid search should try for one hyperparameter.

    {values: [...]} -> those values. {low, high, scale, num} -> `num` evenly spaced points
    from low to high inclusive. On a log scale they should be evenly spaced in log(value),
    e.g. 0.01, 0.1, 1, 10.
    """
    raise NotImplementedError("TASK 2b: list the grid values for spec")


def candidates(cfg, rng) -> list[dict]:
    """The configurations to try, as dicts like {"weight": 0.3, "num_steps": 100, ...}."""
    space = cfg.space
    if cfg.search.strategy == "random":
        return [{k: sample_value(spec, rng) for k, spec in space.items()} for _ in range(cfg.search.trials)]
    if cfg.search.strategy == "grid":
        combos = itertools.product(*(grid_values(spec) for spec in space.values()))
        return [dict(zip(space.keys(), combo)) for combo in combos]
    raise ValueError(f"Unknown search strategy: {cfg.search.strategy}")


def objective(rows: list[dict], cfg) -> float:
    """TASK 3: turn the per-image metrics into ONE number. Lower must be better.

    `rows` has one dict per validation image with the keys of metrics.json (psnr_mean,
    lpips_sample, residual_over_noise, seconds, ...). Decisions you need to make:
      - Which metric(s) matter for this task? (cfg.objective.metric, cfg.objective.goal)
      - How do you combine images: the mean, or the worst image?
      - How do you handle a constraint such as cfg.objective.max_seconds_per_image?
        Hint: returning math.inf makes a configuration lose every comparison.
    Write down the reasons for your choices: you will need them for the report.
    """
    raise NotImplementedError("TASK 3: define the objective")


def evaluate(cfg, params, names, denoiser, device, n_samples, seed=None):
    """Run DPS with sampler settings `params` on the named images; one metrics row per image."""
    override = {"sampler": params | {"n_samples": n_samples}}
    if seed is not None:
        override["seed"] = seed
    return reconstruct_all(OmegaConf.merge(cfg, override), names, denoiser, device)


def confirm_top_k(trials, cfg, denoiser, device) -> tuple[dict, list[dict]]:
    """TASK 4: choose the final configuration more carefully than "the best trial".

    During the search each trial used one seed and few samples, so its objective is noisy:
    with many trials, the best one is partly the luckiest one. Re-evaluate the
    cfg.confirm.top_k best trials on the validation images, once per seed in
    cfg.confirm.seeds (use evaluate(..., seed=s) with cfg.confirm.n_samples), and
    choose the configuration with the best MEAN objective.

    Return (best_params, summary), where summary has one dict per re-checked configuration,
    e.g. {"params": ..., "search_objective": ..., "mean": ..., "std": ...}.
    Does the order of the top trials change when you re-check them?
    """
    # Starting point: simply take the best trial from the search. Replace this.
    best = min(trials, key=lambda t: t["objective"])
    return {k: best[k] for k in cfg.space}, []


def plot_trials(trials, keys, path):
    """Objective against each hyperparameter: which settings actually matter?"""
    finite = [t for t in trials if math.isfinite(t["objective"])]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3), squeeze=False)
    for ax, key in zip(axes[0], keys):
        values = [t[key] for t in finite]
        if all(isinstance(v, str) for v in values):
            labels = sorted(set(values))
            values = [labels.index(v) for v in values]
            ax.set_xticks(range(len(labels)), labels)
        ax.scatter(values, [t["objective"] for t in finite])
        if key == "weight":
            ax.set_xscale("log")
        ax.set_xlabel(key)
        ax.set_ylabel("objective (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mean_metrics(rows):
    return {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k != "image"}


@hydra.main(version_base=None, config_path="../../conf", config_name="optimise")
def main(cfg: DictConfig):
    assert not set(cfg.validation) & set(cfg.test), "validation and test images must not overlap"
    device = device_from(cfg.device)
    denoiser = load_denoiser(cfg.model.name, device)
    out = Path.cwd()
    keys = list(cfg.space)

    # 1-3: search.
    configs = candidates(cfg, random.Random(cfg.seed))
    print(f"Searching {len(configs)} configurations of {keys}")
    trials = []
    for i, params in enumerate(configs):
        rows = evaluate(cfg, params, cfg.validation, denoiser, device, cfg.search.n_samples)
        trials.append(params | {"objective": objective(rows, cfg)} | mean_metrics(rows))
        print(f"trial {i}: {params} -> objective {trials[-1]['objective']:.4f}")
        with open(out / "trials.csv", "w", newline="") as f:  # rewritten each trial: safe to stop early
            writer = csv.DictWriter(f, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
    plot_trials(trials, keys, out / "trials.png")

    # 4: choose.
    best_params, confirmed = confirm_top_k(trials, cfg, denoiser, device)
    print(f"\nChosen on validation: {best_params}")

    # 5: report on held-out test images, never used for any choice above.
    default_params = {k: cfg.sampler[k] for k in keys}
    report = {"chosen": dict(best_params), "default": default_params, "confirm": confirmed, "test": {}}
    for label, params in [("default", default_params), ("chosen", best_params)]:
        rows = evaluate(cfg, params, cfg.test, denoiser, device, cfg.sampler.n_samples)
        report["test"][label] = {"objective": objective(rows, cfg)} | mean_metrics(rows)
        print(f"{label} on test: objective={report['test'][label]['objective']:.4f}")
    (out / "report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print(f"Wrote trials.csv, trials.png and report.yaml to {out}")


if __name__ == "__main__":
    main()
