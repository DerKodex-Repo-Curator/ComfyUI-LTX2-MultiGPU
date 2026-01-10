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
import sys

logger = logging.getLogger("LTXV2MultiGPU")

# ============================================================================
# Detect if ComfyUI-MultiGPU is installed and use its infrastructure
# ============================================================================

_USING_MULTIGPU = False

# Check for ComfyUI-MultiGPU module
for module_name in list(sys.modules.keys()):
    if 'comfyui-multigpu' in module_name.lower() or module_name.startswith('custom_nodes.ComfyUI-MultiGPU'):
        _USING_MULTIGPU = True
        break

# Also try direct import
if not _USING_MULTIGPU:
    try:
        # Try importing from custom_nodes path patterns
        import importlib.util
        import os

        # Check common custom_nodes locations
        possible_paths = []
        for path in sys.path:
            if 'custom_nodes' in path:
                multigpu_path = os.path.join(os.path.dirname(path), 'ComfyUI-MultiGPU', '__init__.py')
                if os.path.exists(multigpu_path):
                    _USING_MULTIGPU = True
                    break
                # Also check sibling
                multigpu_path = os.path.join(path, '..', 'ComfyUI-MultiGPU', '__init__.py')
                if os.path.exists(os.path.normpath(multigpu_path)):
                    _USING_MULTIGPU = True
                    break
    except Exception:
        pass

if _USING_MULTIGPU:
    # ========================================================================
    # USE COMFYUI-MULTIGPU INFRASTRUCTURE
    # ========================================================================
    logger.info("[LTXV2 MultiGPU] ComfyUI-MultiGPU detected, using its device management")

    try:
        # Import from ComfyUI-MultiGPU
        from custom_nodes import ComfyUI_MultiGPU as multigpu

        # Use their wrappers and device functions
        override_class = multigpu.override_class
        override_class_clip = multigpu.override_class_clip
        get_device_list = multigpu.get_device_list

        logger.info("[LTXV2 MultiGPU] Successfully imported ComfyUI-MultiGPU functions")
    except ImportError:
        # Fallback: try alternate import path
        try:
            # Some installations use different module names
            for mod_name, mod in sys.modules.items():
                if 'multigpu' in mod_name.lower() and hasattr(mod, 'override_class'):
                    override_class = mod.override_class
                    override_class_clip = mod.override_class_clip
                    get_device_list = mod.get_device_list
                    logger.info(f"[LTXV2 MultiGPU] Found ComfyUI-MultiGPU via {mod_name}")
                    break
            else:
                raise ImportError("Could not find ComfyUI-MultiGPU exports")
        except Exception as e:
            logger.warning(f"[LTXV2 MultiGPU] Failed to import from ComfyUI-MultiGPU: {e}")
            logger.warning("[LTXV2 MultiGPU] Falling back to standalone mode")
            _USING_MULTIGPU = False

if not _USING_MULTIGPU:
    # ========================================================================
    # STANDALONE MODE - Apply our own patches
    # ========================================================================
    logger.info("[LTXV2 MultiGPU] Running in standalone mode")

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

    # Initialize device state from ComfyUI defaults
    _initial_device = mm.get_torch_device()
    _initial_text_encoder_device = mm.text_encoder_device()

    set_current_device(_initial_device)
    set_current_text_encoder_device(_initial_text_encoder_device)

    # Store original functions
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
# Import nodes (always needed)
# ============================================================================

from .nodes import (
    LTXV2AudioVAELoader,
    LTXV2AVTextEncoderLoader,
    LatentUpscaleModelLoader,
    LTXV2CheckpointLoader,
)

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
if not _USING_MULTIGPU:
    logger.info(f"[LTXV2 MultiGPU] Available devices: {get_device_list()}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
