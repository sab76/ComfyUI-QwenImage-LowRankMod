"""Compress Qwen-Image AdaLN modulation into a shared low-rank projection plus per-layer heads.

Run with ComfyUI's Python:
  python_embeded\\python.exe convert.py INPUT.safetensors --eval-ranks 64,128,256
  python_embeded\\python.exe convert.py INPUT.safetensors --rank 128 --factor-dtype fp32 --out OUTPUT.safetensors
Apply factors fitted on the bf16 model to another copy of it, such as an int8 convrot one:
  python_embeded\\python.exe convert.py QUANTIZED.safetensors --factors OUTPUT.safetensors --out QUANTIZED_LOWRANK.safetensors
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

import comfy.ops
from comfy.ldm.qwen_image.model import QwenTimestepProjEmbeddings

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


class SafeTensorsFile:
    # Plain reads instead of safe_open: on Windows its memory map keeps every touched page resident until exit.
    def __init__(self, path):
        self.file = open(path, "rb")
        n = struct.unpack("<Q", self.file.read(8))[0]
        self.header = json.loads(self.file.read(n))
        self.metadata = self.header.pop("__metadata__", {})
        self.base = 8 + n

    def keys(self):
        return self.header.keys()

    def shape(self, key):
        return self.header[key]["shape"]

    def get_tensor(self, key):
        h = self.header[key]
        start, end = h["data_offsets"]
        buf = bytearray(end - start)
        self.file.seek(self.base + start)
        self.file.readinto(buf)
        return torch.frombuffer(buf, dtype=ST_DTYPES[h["dtype"]]).reshape(h["shape"])

    def copy_to(self, out, key, chunk=64 << 20):
        start, end = self.header[key]["data_offsets"]
        self.file.seek(self.base + start)
        while start < end:
            data = self.file.read(min(chunk, end - start))
            out.write(data)
            start += len(data)


def modulation_layers(f):
    blocks = sorted({int(k.split(".")[1]) for k in f.keys() if k.startswith("transformer_blocks.")})
    return [f"transformer_blocks.{i}.{s}_mod.1" for i in blocks for s in ("img", "txt")] + ["norm_out.linear"]


def timestep_features(f, t, device):
    # silu(temb) exactly as the bf16 model computes it before every modulation linear
    dim = f.shape("time_text_embed.timestep_embedder.linear_2.weight")[0]
    emb = QwenTimestepProjEmbeddings(dim, 0, dtype=torch.bfloat16, device=device, operations=comfy.ops.disable_weight_init)
    emb.load_state_dict({k[len("time_text_embed."):]: f.get_tensor(k) for k in f.keys() if k.startswith("time_text_embed.")})
    return torch.nn.functional.silu(emb(t.to(device), torch.empty(0, dtype=torch.bfloat16, device=device))).detach()


def fit_basis(f, layers, x_cal, device):
    x = x_cal.double().cpu()
    mu = x.mean(0)
    xc = x - mu
    e, q = torch.linalg.eigh(xc.T @ xc / xc.shape[0])
    e = e.clamp(min=e.max().item() * 1e-10)
    s_half = (q * e.sqrt()) @ q.T
    s_ihalf = (q / e.sqrt()) @ q.T

    m = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
    for i, k in enumerate(layers):
        w = f.get_tensor(k + ".weight").to(device, torch.float32)
        m += (w.T @ w).double().cpu()
        print(f"\rgram {i + 1}/{len(layers)}", end="", flush=True)
    print()
    ev, z = torch.linalg.eigh(s_half @ m @ s_half)
    ev, z = ev.flip(0), z.flip(1)
    return mu, s_ihalf, z, ev


def make_down(mu, s_ihalf, z, rank, dtype):
    p = z[:, :rank].T @ s_ihalf
    return p.to(dtype), (-p @ mu).to(dtype)


def linear(x, w, b):
    return torch.nn.functional.linear(x, w.to(x.device), b.to(x.device))


def fit_head(feat, y):
    a = torch.cat([feat.double(), torch.ones(feat.shape[0], 1, dtype=torch.float64, device=feat.device)], 1)
    sol = torch.linalg.lstsq(a.cpu(), y.cpu().double()).solution
    return sol[:-1].T.contiguous(), sol[-1].contiguous()


def evaluate(f, layers, x_cal, x_test, t_test, basis, ranks, dtype, device, keep_rank=None):
    mu, s_ihalf, z, _ = basis
    downs = {r: make_down(mu, s_ihalf, z, r, dtype) for r in ranks}
    feats = {r: (linear(x_cal.to(dtype), *downs[r]), linear(x_test.to(dtype), *downs[r])) for r in ranks}
    zero = (t_test == 0).nonzero()[0, 0].item()
    stats = {r: {"err": 0.0, "var": 0.0, "worst": 0.0, "zero": 0.0} for r in ranks + ["bf16 dense"]}
    heads = {}
    for i, k in enumerate(layers):
        w = f.get_tensor(k + ".weight").to(device)
        b = f.get_tensor(k + ".bias").to(device)
        y_cal = linear(x_cal.float(), w.float(), b.float())
        y_test = linear(x_test.float(), w.float(), b.float())
        y_var = (y_test - y_test.mean(0)).square().sum().item()
        outs = {"bf16 dense": linear(x_test, w, b).float()}
        for r in ranks:
            hw, hb = fit_head(feats[r][0], y_cal)
            hw, hb = hw.to(dtype), hb.to(dtype)
            outs[r] = linear(feats[r][1], hw, hb).to(torch.bfloat16).float()
            if r == keep_rank:
                heads[k] = (hw, hb)
        for r, out in outs.items():
            d = (out - y_test).square()
            st = stats[r]
            st["err"] += d.sum().item()
            st["var"] += y_var
            st["worst"] = max(st["worst"], (d.sum().item() / y_var) ** 0.5)
            st["zero"] = max(st["zero"], (d[zero].sum() / y_test[zero].square().sum()).sqrt().item())
        print(f"\reval {i + 1}/{len(layers)}", end="", flush=True)
    print()
    for r, st in stats.items():
        print(f"rank {r:>10} {('' if r == 'bf16 dense' else args.factor_dtype):>4}: error/variation {100 * (st['err'] / st['var']) ** 0.5:.3f}%  worst layer {100 * st['worst']:.3f}%  t=0 rel {100 * st['zero']:.4f}%")
    return downs.get(keep_rank), heads


def write_safetensors(path, f, layers, replaced, metadata):
    # every key of the modulation layers (including quantization scales) is dropped, replaced tensors are written from memory
    # and everything else is copied from f unchanged
    header = {"__metadata__": dict(f.metadata, **metadata)}
    offset = 0
    keys = [k for k in f.keys() if k.rsplit(".", 1)[0] not in layers] + list(replaced)
    for k in keys:
        if k in replaced:
            t = replaced[k]
            entry = {"dtype": {v: s for s, v in ST_DTYPES.items()}[t.dtype], "shape": list(t.shape)}
            n = t.numel() * t.element_size()
        else:
            entry = {"dtype": f.header[k]["dtype"], "shape": f.shape(k)}
            start, end = f.header[k]["data_offsets"]
            n = end - start
        header[k] = dict(entry, data_offsets=[offset, offset + n])
        offset += n
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(h)))
        out.write(h)
        for i, k in enumerate(keys):
            if k in replaced:
                out.write(replaced[k].contiguous().view(torch.uint8).numpy())
            else:
                f.copy_to(out, k)
            print(f"\rwrite {i + 1}/{len(keys)}", end="", flush=True)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--eval-ranks", default="64,128,192,256,384")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--out")
    parser.add_argument("--factor-dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--calibration", type=int, default=2049)
    parser.add_argument("--factors", help="converted file to take the fitted factors from instead of fitting")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = DTYPES[args.factor_dtype]

    f = SafeTensorsFile(args.input)
    if "time_text_embed.addition_t_embedding.weight" in f.keys():
        raise ValueError("Qwen-Image-Layered timestep conditioning is not supported by this converter.")
    layers = modulation_layers(f)

    if args.factors:
        src = SafeTensorsFile(args.factors)
        factor_keys = ["time_text_embed.modulation_down.weight", "time_text_embed.modulation_down.bias"] + [k + p for k in layers for p in (".weight", ".bias")]
        write_safetensors(args.out, f, set(layers), {k: src.get_tensor(k) for k in factor_keys}, {"modulation_rank": src.metadata["modulation_rank"]})
        sys.exit()

    if any(f.header[k + ".weight"]["dtype"] not in ST_DTYPES for k in layers):
        raise ValueError("Modulation weights are quantized: fit on the bf16 model and apply the result with --factors.")
    t_cal = torch.linspace(0, 1, args.calibration)
    t_test = torch.cat([torch.zeros(1), torch.rand(1024, generator=torch.Generator().manual_seed(0))])
    x_cal = timestep_features(f, t_cal, device)
    x_test = timestep_features(f, t_test, device)

    basis = fit_basis(f, layers, x_cal, device)
    ranks = [args.rank] if args.rank else [int(r) for r in args.eval_ranks.split(",")]
    down, heads = evaluate(f, layers, x_cal, x_test, t_test, basis, ranks, dtype, device, keep_rank=args.rank)

    if args.out:
        replaced = {"time_text_embed.modulation_down.weight": down[0].cpu(), "time_text_embed.modulation_down.bias": down[1].cpu()}
        for k, (hw, hb) in heads.items():
            replaced[k + ".weight"] = hw.cpu()
            replaced[k + ".bias"] = hb.cpu()
        write_safetensors(args.out, f, set(layers), replaced, {"modulation_rank": str(args.rank)})
