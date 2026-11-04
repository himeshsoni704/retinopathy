"""Reproducible retinal preprocessing and controlled image corruption."""
import hashlib
import io

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import convolve, gaussian_filter

FAMILIES = ("defocus", "motion", "vignette", "exposure", "contrast", "haze", "spots", "noise_compression")
OPERATORS = {"defocus": ("defocus",), "motion": ("motion",), "vignette": ("vignette",),
             "exposure": ("dark", "bright"), "contrast": ("contrast",), "haze": ("haze",),
             "spots": ("spots",), "noise_compression": ("noise", "jpeg")}
FAMILY_OF = {op: family for family, ops in OPERATORS.items() for op in ops}


def seed_for(*parts):
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def prepare(path, size):
    with Image.open(path) as source:
        image = source.convert("RGB")
    arr = np.asarray(image)
    field = arr.max(axis=2) > 12
    ys, xs = np.where(field)
    if not len(xs):
        raise ValueError(f"No visible field in {path}")
    image = image.crop((xs.min().item(), ys.min().item(), xs.max().item()+1, ys.max().item()+1))
    side = max(image.size)
    canvas = Image.new("RGB", (side, side))
    canvas.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def degrade(image, op, level, rng):
    if op not in FAMILY_OF or level not in range(1, 6):
        raise ValueError(f"Unknown distortion {op}, level {level}")
    i = level - 1
    arr = np.asarray(image, dtype=np.float32)/255
    mask = arr.max(axis=2) > (12/255)
    if not mask.any():
        raise ValueError("No visible retinal field")
    h, w = mask.shape
    y, x = np.mgrid[:h, :w]
    if op == "defocus":
        out = gaussian_filter(arr, ((1, 2, 3, 4, 6)[i],)*2+(0,))
    elif op == "motion":
        length = (3, 7, 11, 15, 21)[i]
        angle = rng.uniform(0, np.pi)
        offsets = np.linspace(-(length-1)/2, (length-1)/2, length*4)
        kernel = np.zeros((length, length), dtype=np.float32)
        rows = np.rint(offsets*np.sin(angle)).astype(int)+length//2
        cols = np.rint(offsets*np.cos(angle)).astype(int)+length//2
        np.add.at(kernel, (rows, cols), 1)
        out = convolve(arr, (kernel/kernel.sum())[:, :, None], mode="nearest")
    elif op == "vignette":
        radius = np.minimum(1, ((x-w/2)/(w/2))**2 + ((y-h/2)/(h/2))**2)
        out = arr*(1-(.10,.20,.35,.50,.65)[i]*radius[..., None])
    elif op == "dark":
        out = arr*(.85,.70,.55,.40,.25)[i]
    elif op == "bright":
        out = arr*(1.10,1.25,1.50,1.75,2.)[i]
    elif op == "contrast":
        mean = arr[mask].mean(axis=0)
        out = mean+(arr-mean)*(.85,.70,.55,.40,.25)[i]
    elif op == "haze":
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur((.5,1,1.5,2,3)[i])))/255
        veil = (.05,.10,.15,.25,.35)[i]
        out = (1-veil)*blurred+veil
    elif op == "spots":
        cy, cx = np.argwhere(mask)[rng.integers(mask.sum())]
        radius = np.sqrt(mask.sum()*(.01,.02,.04,.06,.08)[i]/np.pi)
        alpha = np.clip(1-np.hypot(x-cx, y-cy)/radius, 0, 1)
        out = arr*(1-.9*alpha[..., None])
    elif op == "noise":
        out = arr+rng.normal(0,(.01,.02,.04,.06,.10)[i],arr.shape)
    else:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=(90,70,50,30,10)[i])
        buf.seek(0)
        with Image.open(buf) as compressed:
            out = np.asarray(compressed.convert("RGB"))/255
    values = np.where(mask[..., None], np.clip(out,0,1), arr)
    return Image.fromarray(np.rint(values*255).astype("uint8"))


def sample_spec(rng, mixture=False, exclude=None):
    families = [f for f in FAMILIES if f != exclude]
    count = int(rng.integers(2,4)) if mixture else 1
    return [(str(rng.choice(OPERATORS[f])), int(rng.integers(1,6)))
            for f in rng.choice(families, count, replace=False)]
