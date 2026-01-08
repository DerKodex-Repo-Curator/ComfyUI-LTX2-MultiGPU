# Dreamfast LTX2 Multi-GPU
# Copyright (C) 2025 Dreamfast
# Multi-GPU device selection for LTXV2 video generation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Device utilities based on ComfyUI-MultiGPU by pollockjj (GPL-3.0)

import logging
import torch
import comfy.model_management as mm

from .device_utils import get_device_list
from .wrappers import (
    override_class,
    override_class_clip,
    get_current_device,
    set_current_device,
    get_current_text_encoder_device,
    set_current_text_encoder_device,
)
from .nodes import (
    LTXV2AudioVAELoader,
    LTXV2AVTextEncoderLoader,
    LatentUpscaleModelLoader,
    LTXV2CheckpointLoader,
)

logger = logging.getLogger("LTXV2MultiGPU")

# ============================================================================
# Initialize device state from ComfyUI defaults
# ============================================================================

_initial_device = mm.get_torch_device()
_initial_text_encoder_device = mm.text_encoder_device()

set_current_device(_initial_device)
set_current_text_encoder_device(_initial_text_encoder_device)

# ============================================================================
# Patch ComfyUI model_management to use our device selection
# ============================================================================

_original_get_torch_device = mm.get_torch_device
_original_text_encoder_device = mm.text_encoder_device


def get_torch_device_patched():
    """Return the user-selected device instead of ComfyUI default."""
    current = get_current_device()
    if current is None:
        return _original_get_torch_device()

    devs = get_device_list()
    current_str = str(current)

    if mm.cpu_state == mm.CPUState.CPU or "cpu" in current_str.lower():
        return torch.device("cpu")

    if current_str in devs:
        return torch.device(current_str)

    return _original_get_torch_device()


def text_encoder_device_patched():
    """Return the user-selected text encoder device."""
    current = get_current_text_encoder_device()
    if current is None:
        return _original_text_encoder_device()

    devs = get_device_list()
    current_str = str(current)

    if current_str in devs:
        return torch.device(current_str)

    return _original_text_encoder_device()


# Apply patches
mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched

logger.info("[LTXV2 MultiGPU] Patched comfy.model_management for device selection")

# ============================================================================
# Register nodes
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LTXV2CheckpointLoaderMultiGPU": override_class(LTXV2CheckpointLoader),
    "LTXV2AudioVAELoaderMultiGPU": override_class(LTXV2AudioVAELoader),
    "LTXV2AVTextEncoderLoaderMultiGPU": override_class_clip(LTXV2AVTextEncoderLoader),
    "LatentUpscaleModelLoaderMultiGPU": override_class(LatentUpscaleModelLoader),
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXV2CheckpointLoaderMultiGPU": "LTXV2 Checkpoint Loader (MultiGPU)",
    "LTXV2AudioVAELoaderMultiGPU": "LTXV2 Audio VAE Loader (MultiGPU)",
    "LTXV2AVTextEncoderLoaderMultiGPU": "LTXV2 Text Encoder Loader (MultiGPU)",
    "LatentUpscaleModelLoaderMultiGPU": "Latent Upscale Model Loader (MultiGPU)",
}

logger.info(f"[LTXV2 MultiGPU] Registered {len(NODE_CLASS_MAPPINGS)} nodes")
logger.info(f"[LTXV2 MultiGPU] Available devices: {get_device_list()}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
