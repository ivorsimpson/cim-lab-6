"""Collect metrics.json files from a sweep into one table (and optionally a plot).

    uv run python -m cim_lab_6.summarise outputs/steps_sweep
    uv run python -m cim_lab_6.summarise outputs/steps_sweep --x sampler.num_steps --y psnr_mean lpips_sample
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= flatten(v, f"{prefix}{k}.")
        else:
            out[f"{prefix}{k}"] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--x", help="config key for the x axis, e.g. sampler.num_steps")
    parser.add_argument("--y", nargs="+", default=["psnr_mean", "lpips_sample"])
    args = parser.parse_args()

    runs = []
    for path in sorted(args.folder.rglob("metrics.json")):
        result = json.loads(path.read_text())
        runs.append(flatten(result["config"]) | result["mean"] | {"run": str(path.parent)})
    if not runs:
        raise SystemExit(f"No metrics.json found under {args.folder}")
    if args.x:
        runs.sort(key=lambda r: r[args.x])

    # Only show the config values that actually change across the sweep.
    keys = [k for k in runs[0] if "." in k and len({str(r.get(k)) for r in runs}) > 1]
    metrics = [k for k in runs[0] if "." not in k and k != "run" and k not in ("seed", "device", "mode", "experiment_name")]
    with open(args.folder / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + metrics)
        writer.writeheader()
        writer.writerows([{k: r.get(k) for k in keys + metrics} for r in runs])

    print(" | ".join(keys + metrics))
    for r in runs:
        print(" | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys + metrics))
    print(f"\nWrote {args.folder / 'summary.csv'}")

    if args.x:
        fig, axes = plt.subplots(1, len(args.y), figsize=(4 * len(args.y), 3.2))
        for ax, metric in zip(axes if len(args.y) > 1 else [axes], args.y):
            ax.plot([r[args.x] for r in runs], [r[metric] for r in runs], "o-")
            ax.set_xlabel(args.x)
            ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(args.folder / "summary.png", dpi=150)
        print(f"Wrote {args.folder / 'summary.png'}")


if __name__ == "__main__":
    main()
