# Dreamfast LTX2 Multi-GPU - Wrapper Functions
# Copyright (C) 2025 Dreamfast
# Based on ComfyUI-MultiGPU by pollockjj (GPL-3.0)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import copy
import logging
from .device_utils import get_device_list

logger = logging.getLogger("LTXV2MultiGPU")

# Global device state
_current_device = None
_current_text_encoder_device = None


def get_current_device():
    """Get the currently selected compute device."""
    global _current_device
    return _current_device


def set_current_device(device):
    """Set the current compute device."""
    global _current_device
    _current_device = device
    logger.debug(f"Set current device to: {device}")


def get_current_text_encoder_device():
    """Get the currently selected text encoder device."""
    global _current_text_encoder_device
    return _current_text_encoder_device


def set_current_text_encoder_device(device):
    """Set the current text encoder device."""
    global _current_text_encoder_device
    _current_text_encoder_device = device
    logger.debug(f"Set text encoder device to: {device}")


def override_class(cls):
    """Add device dropdown to a loader node for UNet/VAE models."""

    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "dreamfast/ltxv2"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            original_device = get_current_device()
            if device is not None:
                set_current_device(device)
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            try:
                return out
            finally:
                set_current_device(original_device)

    return NodeOverride


def override_class_clip(cls):
    """Add device dropdown to a loader node for CLIP/text encoder models."""

    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "dreamfast/ltxv2"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            original_text_device = get_current_text_encoder_device()
            if device is not None:
                set_current_text_encoder_device(device)
            # Pass device='default' to let our patched function handle it
            kwargs['device'] = 'default'
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            try:
                return out
            finally:
                set_current_text_encoder_device(original_text_device)

    return NodeOverride
