from pathlib import Path

import deepinv as dinv
import torch
from hydra.utils import to_absolute_path
from PIL import Image
from torchvision.transforms.functional import center_crop, resize, to_tensor


def square(image: torch.Tensor, size: int) -> torch.Tensor:
    """Centre-crop to a square and resize, keeping 3 channels."""
    image = image.expand(3, -1, -1) if image.shape[0] == 1 else image[:3]
    side = min(image.shape[-2:])
    return resize(center_crop(image, side), [size, size], antialias=True).clamp(0, 1)


def load_image(name: str, root: str, size: int) -> torch.Tensor:
    """Load one image as a (1, 3, size, size) tensor in [0, 1].

    `name` is either a file stem relative to `root` (e.g. faces/face_00), or
    dinv:<file> for a DeepInverse example image (downloaded and cached on first use).
    """
    if name.startswith("dinv:"):
        image = dinv.utils.load_example(name.removeprefix("dinv:"))[0]
        return square(image, size).unsqueeze(0)

    base = Path(to_absolute_path(root))
    stem = (base / name).resolve()
    if base.resolve() not in stem.parents:
        raise ValueError("Images must be inside data.root.")
    for path in [stem, *(stem.with_name(stem.name + e) for e in (".png", ".jpg", ".jpeg"))]:
        if path.is_file():
            with Image.open(path) as image:
                return square(to_tensor(image.convert("RGB")), size).unsqueeze(0)
    raise FileNotFoundError(f"Image '{name}' was not found in {base}.")
