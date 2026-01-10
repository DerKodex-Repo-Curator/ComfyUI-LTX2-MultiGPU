# LTX2 Multi-GPU

Multi-GPU device selection for LTXV2 video generation in ComfyUI.

## Features

- Select which GPU to load LTXV2 models on
- Combined checkpoint loader for video model + video VAE + audio VAE
- Works with LTXV2 audio-video generation workflows

## Notes

I had OOM issues even with a 4090+5090 setup. The main issue was the text encoder and originally not being able to assign different nodes to different GPUS or CPU. Having a decent CPU and offloading the text encoder helped a lot.

With this I was able to generate 10 second 1080p videos by offloading the text encoder to CPU and the final vae to `cuda:0` with the rest on `cuda:1`.

The workflow provided in `workflows/multi-gpu-ltx2-t2v.json` contains an ollama prompt enhancer and the use of this node to generate 720p videos. It's based off the original t2v workflow provided by comfyui. I had an error with audio encoding and that seems to be resolved using the `Video Combine` node with a latest version of `ffpmeg`. It may need some adjustments or you can just use these nodes to replace your current LTXv nodes in your current workflow.

The preview in the `Video Combine` node seems buggy but I've been able to right click and `Open Preview` to see the result, despite it not being there in the node.

## Nodes

| Node | Description |
|------|-------------|
| **LTXV2 Checkpoint Loader (MultiGPU)** | Loads video model, video VAE, and audio VAE from a single LTXV2 checkpoint |
| **LTXV2 Audio VAE Loader (MultiGPU)** | Loads audio VAE for audio encoding/decoding |
| **LTXV2 Text Encoder Loader (MultiGPU)** | Loads Gemma 3 text encoder for LTXV2 |
| **Latent Upscale Model Loader (MultiGPU)** | Loads latent upscale model for video upsampling |

<img width="345" height="364" alt="image" src="https://github.com/user-attachments/assets/a92760b8-cf24-45b8-b7a3-845dd4af76ae" />
<img width="361" height="119" alt="image" src="https://github.com/user-attachments/assets/3327c245-59e3-46e2-8df1-aaf9dadd4bb0" />

## Installation

Clone or copy this folder to your ComfyUI `custom_nodes` directory:

```
cd ComfyUI/custom_nodes/
git clone https://github.com/dreamfast/ComfyUI-LTX2-MultiGPU
```

Restart ComfyUI.

## Usage

Each node has a `device` dropdown that lets you select which GPU to load the model on:
- `cpu` - Load on CPU (slow but saves VRAM)
- `cuda:0` - First GPU
- `cuda:1` - Second GPU
- etc.

## Requirements

- ComfyUI with LTXV2 support (built-in `LTXVAudioVAELoader`, `LTXAVTextEncoderLoader` nodes)
- PyTorch with CUDA support (for multi-GPU)

No `requirements.txt` is needed - this package only uses dependencies already included with ComfyUI (torch, comfy.model_management, standard library).

## Compatibility with ComfyUI-MultiGPU

This package is designed to work alongside [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) without conflicts:

- **If ComfyUI-MultiGPU is installed**: Uses its device management infrastructure (no duplicate patches)
- **If standalone**: Applies its own patches to `comfy.model_management`

You can use both packages together - check the ComfyUI logs on startup for:
- `[LTXV2 MultiGPU] ComfyUI-MultiGPU detected, using its device management`
- `[LTXV2 MultiGPU] Running in standalone mode`

## License

GPL-3.0 - See LICENSE file.

## Attribution

Device utilities based on [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) by pollockjj.
