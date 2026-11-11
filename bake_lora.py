"""Bake a LoRA into an int8 convrot (low-rank AdaLN) Qwen-Image checkpoint.

Merged weights are quantized from base_bf16 + strength * alpha/rank * up @ down in fp32,
with the same eager comfy_kitchen convrot quantizer that reproduces the template exactly.
All other tensors and the header are copied byte for byte from the template.
"""
import argparse, json, struct
import torch
from safetensors import safe_open
from comfy_kitchen.backends.eager import quantization as eq

p = argparse.ArgumentParser()
p.add_argument("base_bf16")
p.add_argument("template")
p.add_argument("lora")
p.add_argument("out")
p.add_argument("--strength", type=float, default=1.0)
args = p.parse_args()

with open(args.template, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    header_bytes = f.read(hlen)
header = json.loads(header_bytes)
tensors = sorted(((k, v) for k, v in header.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0])

with safe_open(args.lora, "pt") as lf:
    lora_layers = sorted({k.rsplit(".lora_", 1)[0] for k in lf.keys() if ".lora_" in k})
for layer in lora_layers:
    conf = header.get(layer + ".comfy_quant")
    assert conf is not None and layer + ".weight_scale" in header, f"{layer} is not an int8 convrot layer in the template"

pending = {}
done = 0
with safe_open(args.base_bf16, "pt", device="cuda") as bf, safe_open(args.lora, "pt", device="cuda") as lf, \
        open(args.template, "rb") as src, open(args.out, "wb") as dst:
    tq = safe_open(args.template, "pt")
    group_sizes = {}
    for layer in lora_layers:
        group_sizes[layer] = json.loads(bytes(tq.get_tensor(layer + ".comfy_quant").tolist()))["convrot_groupsize"]
    del tq

    dst.write(struct.pack("<Q", hlen))
    dst.write(header_bytes)
    data_start = 8 + hlen
    lora_set = set(lora_layers)
    for name, info in tensors:
        start, end = info["data_offsets"]
        layer, _, suffix = name.rpartition(".")
        if layer in lora_set and suffix in ("weight", "weight_scale"):
            if name not in pending:
                w = bf.get_tensor(layer + ".weight").float()
                down = lf.get_tensor(layer + ".lora_down.weight").float()
                up = lf.get_tensor(layer + ".lora_up.weight").float()
                alpha = lf.get_tensor(layer + ".alpha").item()
                w.addmm_(up, down, alpha=args.strength * alpha / down.shape[0])
                q, s = eq.quantize_int8_convrot_weight(w, group_sizes[layer])
                pending[layer + ".weight"] = q.cpu().numpy().tobytes()
                pending[layer + ".weight_scale"] = s.cpu().numpy().tobytes()
                done += 1
                if done % 60 == 0:
                    print(f"merged {done}/{len(lora_layers)} layers", flush=True)
            data = pending.pop(name)
            assert len(data) == end - start, name
            dst.write(data)
        else:
            src.seek(data_start + start)
            remaining = end - start
            while remaining:
                chunk = src.read(min(remaining, 64 << 20))
                dst.write(chunk)
                remaining -= len(chunk)
    assert not pending and done == len(lora_layers)
print(f"wrote {args.out}: merged {done} layers at strength {args.strength}")
