"""Run with ComfyUI's Python: verify.py ORIGINAL.safetensors CONVERTED.safetensors"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

import comfy.model_management as mm
import comfy.sample
import comfy.sd

before = comfy.sd.load_diffusion_model
spec = importlib.util.spec_from_file_location("lowrank_mod", os.path.join(os.path.dirname(__file__), "__init__.py"), submodule_search_locations=[os.path.dirname(__file__)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
assert comfy.sd.load_diffusion_model is before
from lowrank_mod.nodes import load_low_rank_qwen_image
from lowrank_mod.convert import SafeTensorsFile

patcher = load_low_rank_qwen_image(sys.argv[2])
model = patcher.model.diffusion_model

latent = torch.zeros(1, 16, 1, 32, 32)
context = torch.randn(1, 24, 3584)
cond = [[context, {"reference_latents": [torch.randn(1, 16, 1, 32, 32)], "reference_latents_method": "index_timestep_zero"}]]
result = comfy.sample.sample(patcher, comfy.sample.prepare_noise(latent, 0), 4, 1.0, "euler", "simple", cond, [[torch.zeros_like(context), {}]], latent, seed=0)
assert torch.isfinite(result).all()
mm.unload_all_models()

factors = [p for n, p in model.named_parameters() if "modulation_down" in n or "_mod.1." in n or "norm_out.linear" in n]
assert all(p.dtype == torch.float32 for p in factors), "modulation factors lost fp32"

f = SafeTensorsFile(sys.argv[1])
t = torch.cat([torch.zeros(1), torch.rand(64)])
x = model.time_text_embed.timestep_embedder(model.time_text_embed.time_proj(t).to(torch.bfloat16))
feat = model.time_text_embed(t, x)
worst = 0.0
for k in ["transformer_blocks.0.img_mod.1", "transformer_blocks.30.txt_mod.1", "transformer_blocks.59.img_mod.1", "norm_out.linear"]:
    ref = torch.nn.functional.linear(torch.nn.functional.silu(x), f.get_tensor(k + ".weight"), f.get_tensor(k + ".bias")).float()
    out = model.get_submodule(k)(feat).float()
    rel = ((out - ref).norm() / (ref - ref.mean(0)).norm()).item()
    worst = max(worst, rel)
    print(f"{k}: error/variation vs dense bf16 {100 * rel:.3f}%")
assert worst < 0.03
print("PASS: strict load, sampling with timestep-zero refs, fp32 factors kept, matches dense modulation, built-in loader untouched")
