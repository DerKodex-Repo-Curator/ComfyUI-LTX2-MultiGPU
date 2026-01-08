# Dreamfast LTX2 Multi-GPU - Device Utilities
# Copyright (C) 2025 Dreamfast
# Based on ComfyUI-MultiGPU by pollockjj (GPL-3.0)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import torch
import logging
import gc

logger = logging.getLogger("LTXV2MultiGPU")

_DEVICE_LIST_CACHE = None


def get_device_list():
    """
    Enumerate all available devices that can store torch tensors.
    Results are cached after first call since devices don't change during runtime.

    Returns list like: ["cpu", "cuda:0", "cuda:1", "mps", ...]
    """
    global _DEVICE_LIST_CACHE

    if _DEVICE_LIST_CACHE is not None:
        return _DEVICE_LIST_CACHE

    devs = ["cpu"]

    # CUDA (NVIDIA + AMD ROCm)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devs += [f"cuda:{i}" for i in range(device_count)]
        logger.debug(f"Found {device_count} CUDA device(s)")

    # XPU (Intel)
    try:
        import intel_extension_for_pytorch
    except ImportError:
        pass

    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        device_count = torch.xpu.device_count()
        devs += [f"xpu:{i}" for i in range(device_count)]
        logger.debug(f"Found {device_count} XPU device(s)")

    # MPS (Apple Metal)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
        logger.debug("Found MPS device")

    # DirectML (Windows)
    try:
        import torch_directml
        adapter_count = torch_directml.device_count()
        if adapter_count > 0:
            devs += [f"directml:{i}" for i in range(adapter_count)]
            logger.debug(f"Found {adapter_count} DirectML adapter(s)")
    except ImportError:
        pass

    _DEVICE_LIST_CACHE = devs
    logger.debug(f"Device list: {devs}")

    return devs


def soft_empty_cache_all_devices():
    """Clear allocator caches across all devices."""
    gc.collect()

    all_devices = get_device_list()
    is_cuda_available = hasattr(torch, "cuda") and torch.cuda.is_available()

    for device_str in all_devices:
        if device_str.startswith("cuda:") and is_cuda_available:
            device_idx = int(device_str.split(":")[1])
            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

        elif device_str == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        elif device_str.startswith("xpu:"):
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
                torch.xpu.empty_cache()
