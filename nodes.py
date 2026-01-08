# Dreamfast LTX2 Multi-GPU - Node Definitions
# Copyright (C) 2025 Dreamfast
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import folder_paths
from nodes import NODE_CLASS_MAPPINGS


def _convert_node_output(result):
    """Convert io.NodeOutput to tuple for ComfyUI compatibility.

    io.NodeOutput from comfy_api.latest stores values in self.args tuple.
    """
    if isinstance(result, tuple):
        return result

    # NodeOutput stores values in self.args
    if type(result).__name__ == 'NodeOutput' and hasattr(result, 'args'):
        return result.args

    # Fallback for other iterables
    if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
        return tuple(result)

    return (result,)


class LTXV2AudioVAELoader:
    """Adapter for ComfyUI core LTXV2 audio VAE loader."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),
                             {"tooltip": "LTXV2 checkpoint containing audio VAE"})
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("audio_vae",)
    FUNCTION = "load"
    CATEGORY = "dreamfast/ltxv2"
    TITLE = "LTXV2 Audio VAE Loader"

    def load(self, ckpt_name):
        """Load LTXV2 audio VAE from checkpoint."""
        core_node = NODE_CLASS_MAPPINGS.get("LTXVAudioVAELoader")
        if core_node is None:
            raise RuntimeError("LTXVAudioVAELoader not found in ComfyUI core nodes.")

        result = core_node.execute(ckpt_name=ckpt_name)
        return _convert_node_output(result)


class LTXV2AVTextEncoderLoader:
    """Adapter for ComfyUI core LTXV2 audio-video text encoder loader (Gemma 3)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),
                                {"tooltip": "Text encoder model file (Gemma 3)"}),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),
                             {"tooltip": "LTXV2 checkpoint for text encoder config"})
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "dreamfast/ltxv2"
    TITLE = "LTXV2 Text Encoder Loader"

    def load(self, text_encoder, ckpt_name, device=None):
        """Load LTXV2 audio-video text encoder (Gemma 3)."""
        core_node = NODE_CLASS_MAPPINGS.get("LTXAVTextEncoderLoader")
        if core_node is None:
            raise RuntimeError("LTXAVTextEncoderLoader not found in ComfyUI core nodes.")

        result = core_node.execute(text_encoder=text_encoder, ckpt_name=ckpt_name)
        return _convert_node_output(result)


class LatentUpscaleModelLoader:
    """Adapter for ComfyUI core Latent Upscale Model Loader."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("latent_upscale_models"),
                              {"tooltip": "Latent upscale model"})
            }
        }

    RETURN_TYPES = ("LATENT_UPSCALE_MODEL",)
    FUNCTION = "load"
    CATEGORY = "dreamfast/ltxv2"
    TITLE = "Latent Upscale Model Loader"

    def load(self, model_name):
        """Load latent upscale model for video upsampling."""
        core_node = NODE_CLASS_MAPPINGS.get("LatentUpscaleModelLoader")
        if core_node is None:
            raise RuntimeError("LatentUpscaleModelLoader not found in ComfyUI core nodes.")

        result = core_node.execute(model_name=model_name)
        return _convert_node_output(result)


class LTXV2CheckpointLoader:
    """Combined loader for LTXV2 video model, video VAE, and audio VAE.

    Loads all components needed for LTXV2 audio-video generation from a
    single checkpoint file.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),
                             {"tooltip": "LTXV2 checkpoint (video model + video VAE + audio VAE)"})
            }
        }

    RETURN_TYPES = ("MODEL", "VAE", "VAE")
    RETURN_NAMES = ("model", "vae", "audio_vae")
    FUNCTION = "load"
    CATEGORY = "dreamfast/ltxv2"
    TITLE = "LTXV2 Checkpoint Loader"

    def load(self, ckpt_name):
        """Load LTXV2 video model, video VAE, and audio VAE from one checkpoint."""
        # Load video model and VAE using CheckpointLoaderSimple
        checkpoint_loader = NODE_CLASS_MAPPINGS.get("CheckpointLoaderSimple")
        if checkpoint_loader is None:
            raise RuntimeError("CheckpointLoaderSimple not found in ComfyUI core nodes.")

        model, clip, vae = checkpoint_loader().load_checkpoint(ckpt_name)

        # Load audio VAE from the same checkpoint
        audio_vae_loader = NODE_CLASS_MAPPINGS.get("LTXVAudioVAELoader")
        if audio_vae_loader is None:
            raise RuntimeError("LTXVAudioVAELoader not found in ComfyUI core nodes.")

        audio_vae_result = audio_vae_loader.execute(ckpt_name=ckpt_name)
        audio_vae = _convert_node_output(audio_vae_result)[0]

        return (model, vae, audio_vae)
