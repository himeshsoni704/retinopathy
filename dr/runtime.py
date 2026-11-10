"""Explicit precision, output protection and reproducible runtime reporting."""
import platform
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
import torch


def protect(path, overwrite=False):
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new output or pass --overwrite")


def protect_all(paths, overwrite=False):
    for path in paths:
        protect(path,overwrite)


def precision(device, requested='auto', training=True):
    if device.type != 'cuda':
        if requested in ('fp16', 'bf16'):
            raise ValueError('FP16/BF16 modes require CUDA; use auto or fp32 on CPU')
        return 'fp32'
    result = ('fp16' if training else 'fp32') if requested == 'auto' else requested
    if result == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support BF16')
    return result


def autocast(device, mode):
    return nullcontext() if mode == 'fp32' else torch.autocast('cuda', dtype={'fp16':torch.float16,'bf16':torch.bfloat16}[mode])


def environment(device):
    packages = {}
    for name in ('torch','torchvision','numpy','scipy','pandas','scikit-learn','Pillow'):
        packages[name] = version(name)
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                cuda=torch.version.cuda, deterministic=torch.are_deterministic_algorithms_enabled())


class Profile:
    def __init__(self, device):
        self.device = device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        self.start = time.perf_counter()

    def finish(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return dict(seconds=time.perf_counter()-self.start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None)
