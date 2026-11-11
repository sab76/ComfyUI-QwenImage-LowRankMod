# ComfyUI-QwenImage-LowRankMod

Loader for Qwen-Image checkpoints whose AdaLN modulation layers have been replaced by one shared low-rank projection and small per-layer heads.

In Qwen-Image every block has `img_mod` and `txt_mod` linears (3072 → 18432) that only see `silu(temb)`, which depends on the timestep alone. That is 6.8B of the model's ~20B parameters. The converter fits one shared `3072 → r` projection of `silu(temb)` plus a `r → out` head for each of the 120 block modulations and `norm_out`, calibrated on real timestep embeddings over the full `t ∈ [0, 1]` range including `t = 0`.

## Results (Qwen-Image-Edit-2511, rank 128, fp32 factors)

| | bf16 original | rank 128 |
|---|---|---|
| Modulation params | 6.8B (13.6 GB) | 284M (1.1 GB fp32) |
| File size | 40.9 GB | 28.4 GB |
| File size, int8 convrot | 20.5 GB | 14.8 GB |
| Modulation error vs exact, held-out timesteps | 1.008% | 1.008% |
| Worst layer | 1.094% | 1.094% |
| Relative error at `t = 0` (reference images) | 0.196% | 0.196% |

Errors are relative to how much each layer's output varies with the timestep. At rank 96 and above the compressed modulation is as close to the exact math as the bf16 original is, so the remaining error is bf16 rounding. The factors must stay in fp32: with bf16 factors the error rises to 1.44% at rank 64–192 and grows at higher ranks.

## Requirements

- A recent ComfyUI with native Qwen-Image-Edit-2511 support. Int8/convrot checkpoints also need ComfyUI's mixed-precision quantization ops. Tested on ComfyUI at commit `7dd4627`.
- No extra Python dependencies.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/sab76/ComfyUI-QwenImage-LowRankMod.git
```
Restart ComfyUI.

## Models

Converted checkpoints are on Hugging Face: [sab96/Qwen-Image-Edit-2511-LowRankMod](https://huggingface.co/sab96/Qwen-Image-Edit-2511-LowRankMod).

| File | Size | Notes |
|---|---|---|
| `qwen_image_edit_2511_int8_convrot_lowrankmod_r128.safetensors` | 14.8 GB | int8 convrot |
| `qwen_image_edit_2511_int8_convrot_lowrankmod_r128_lightning8s.safetensors` | 14.8 GB | int8 convrot, Lightning 8-step LoRA baked in |
| `qwen_image_edit_2511_lowrankmod_r128.safetensors` | 28.4 GB | bf16 |

## Usage

1. Put the converted checkpoint in `models/diffusion_models` (or `models/unet`).
2. Replace **Load Diffusion Model** with **Load Qwen-Image (Low-Rank AdaLN)** (`advanced/loaders`). The rest of the workflow is unchanged.

The regular loaders cannot load these files, because the modulation weights have different shapes.

## Converting

Run from the `ComfyUI` folder with ComfyUI's Python. The converter streams the checkpoint and uses a few GB of RAM. It needs a CUDA GPU for reasonable speed.

Evaluate ranks without writing anything:
```
python_embeded\python.exe -s custom_nodes/ComfyUI-QwenImage-LowRankMod/convert.py models/unet/qwen_image_edit_2511_bf16.safetensors --factor-dtype fp32 --eval-ranks 64,96,128
```

Convert the bf16 model:
```
python_embeded\python.exe -s custom_nodes/ComfyUI-QwenImage-LowRankMod/convert.py models/unet/qwen_image_edit_2511_bf16.safetensors --rank 128 --factor-dtype fp32 --out models/unet/qwen_image_edit_2511_lowrankmod_r128.safetensors
```

Apply those factors to a quantized copy of the same model (for example int8 convrot). Fit on bf16 first: quantized modulation weights have already lost precision, so the converter refuses to fit on them. The dense modulation layers and their quantization scales are dropped, and the rest of the file is copied unchanged.
```
python_embeded\python.exe -s custom_nodes/ComfyUI-QwenImage-LowRankMod/convert.py models/unet/qwen_image_edit_2511_int8_convrot.safetensors --factors models/unet/qwen_image_edit_2511_lowrankmod_r128.safetensors --out models/unet/qwen_image_edit_2511_int8_convrot_lowrankmod_r128.safetensors
```

## Baking a LoRA

`bake_lora.py` merges a LoRA into an int8 convrot low-rank checkpoint. This saves the RAM the separate LoRA uses and the cost of applying its patches every step. Each targeted layer is re-quantized from the bf16 weight plus the LoRA in fp32. Everything else is copied unchanged from the int8 file, including the low-rank factors. The file size does not change.
```
python_embeded\python.exe -s custom_nodes/ComfyUI-QwenImage-LowRankMod/bake_lora.py models/unet/qwen_image_edit_2511_bf16.safetensors models/unet/qwen_image_edit_2511_int8_convrot_lowrankmod_r128.safetensors models/loras/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors models/unet/qwen_image_edit_2511_int8_convrot_lowrankmod_r128_lightning8s.safetensors --strength 1.0
```
The bf16 file must be the model the int8 file was quantized from. Only LoRAs on int8 convrot layers (attention and MLP) can be baked. The script stops before writing if the LoRA targets any other layer. Needs a CUDA GPU. Close ComfyUI first to free VRAM and RAM.

## Limitations

- LoRAs targeting `img_mod`, `txt_mod` or `norm_out.linear` do not apply. LoRAs on attention and MLP layers work as usual.
- Qwen-Image-Layered (extra timestep conditioning) is not supported.
- The node subclasses ComfyUI's Qwen-Image model, so large upstream changes to that code may require an update.

## Credits

- Based on [bluvoll/ComfyUI-MageFlow-Compressed](https://github.com/bluvoll/ComfyUI-MageFlow-Compressed), which does the same low-rank AdaLN compression for Mage-Flow.
- Model code adapted from [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- Base model: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511). Lightning LoRA: [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning).

## License

Code: GPL-3.0-or-later, see `LICENSE`. The model weights are distributed separately under their own license (Apache-2.0, from Qwen-Image-Edit-2511).
