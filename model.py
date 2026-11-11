import torch
import comfy.ops
from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel, QwenTimestepProjEmbeddings

# The shared projection and heads are fitted in fp32: bf16 factors add rounding error that grows with rank.
# They always use manual_cast ops, quantized model ops would cast them to the compute dtype on load.
FactorLinear = comfy.ops.manual_cast.Linear


class LowRankTimestepEmbeddings(QwenTimestepProjEmbeddings):
    def __init__(self, modulation_rank, embedding_dim, dtype=None, device=None, operations=None):
        super().__init__(embedding_dim, 0, dtype=dtype, device=device, operations=operations)
        self.modulation_down = FactorLinear(embedding_dim, modulation_rank, dtype=torch.float32, device=device)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        return self.modulation_down(torch.nn.functional.silu(super().forward(timestep, hidden_states)).float())


class LowRankModQwenImageTransformer(QwenImageTransformer2DModel):
    """Qwen-Image with every AdaLN linear fed by one shared projection of silu(temb)."""
    def __init__(self, modulation_rank, compute_dtype, operations, dtype=None, device=None, **kwargs):
        class ModulationHead(FactorLinear):
            def forward(self, x):
                return super().forward(x).to(compute_dtype)

        class LowRankOperations(operations):
            # dim -> 6 * dim block modulation and dim -> 2 * dim final modulation are the only such linears in the model
            @staticmethod
            def Linear(in_features, out_features, bias=True, device=None, dtype=None):
                if out_features in (2 * in_features, 6 * in_features):
                    return ModulationHead(modulation_rank, out_features, bias=bias, device=device, dtype=torch.float32)
                return operations.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

        super().__init__(dtype=dtype, device=device, operations=LowRankOperations, **kwargs)
        self.time_text_embed = LowRankTimestepEmbeddings(modulation_rank, self.inner_dim, dtype=dtype, device=device, operations=operations)
        for block in self.transformer_blocks:
            block.img_mod[0] = torch.nn.Identity()
            block.txt_mod[0] = torch.nn.Identity()
        self.norm_out.silu = torch.nn.Identity()
