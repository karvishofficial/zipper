"""
PyTorch Implementation of Cactus Needle 2 (45M) for RTX 4050 GPU Training.
Loads weights directly from needle2.pkl and executes entirely on cuda:0.
"""

import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA GPU is not available!"
DEVICE = torch.device("cuda:0")
assert "RTX 4050" in torch.cuda.get_device_name(DEVICE), f"Expected RTX 4050, got {torch.cuda.get_device_name(DEVICE)}"


class ZCRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        rms = torch.sqrt(torch.mean(xf ** 2, dim=-1, keepdim=True) + self.eps)
        return ((1.0 + self.scale) * xf / rms).to(x.dtype)


def precompute_rope_freqs(head_dim, seq_len, theta=100000.0, device=DEVICE):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    # x: [B, num_heads, T, head_dim]
    T = x.shape[2]
    half = x.shape[-1] // 2
    c = cos[:T][None, None, :, :]
    s = sin[:T][None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


def _rms_unit(x, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(torch.mean(xf ** 2, dim=-1, keepdim=True) + eps)).to(x.dtype)


def _walsh_matrix(n, device=DEVICE):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return torch.tensor(H / np.sqrt(n), dtype=torch.float32, device=device)


class HadamardMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        n = 1 << (d_model - 1).bit_length()
        self.n = n
        self.register_buffer("H", _walsh_matrix(n))
        self.d1 = nn.Parameter(torch.ones(n))
        self.d2 = nn.Parameter(torch.ones(n))
        self.d3 = nn.Parameter(torch.full((n,), 0.02))

    def forward(self, x):
        pad = self.n - self.d_model
        z = F.pad(x, (0, pad)) if pad else x
        z = (self.d1 * z) @ self.H
        z = F.silu(self.d2 * z) @ self.H
        return (self.d3 * z)[..., :self.d_model]


class LoRALinear(nn.Module):
    def __init__(self, base_weight: torch.Tensor, rank=16, alpha=32.0):
        super().__init__()
        # Flax kernel is [in_features, out_features]
        # In PyTorch, x @ kernel == F.linear(x, weight) where weight is [out_features, in_features]
        in_features, out_features = base_weight.shape
        self.register_buffer("base_weight", base_weight.clone().detach())
        self.rank = rank
        self.scale = alpha / rank
        
        # LoRA A: [in_features, rank], LoRA B: [rank, out_features]
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=base_weight.device) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=base_weight.device))

    def forward(self, x):
        # x: [..., in_features]
        base_out = x @ self.base_weight
        lora_out = (x @ self.lora_A) @ self.lora_B
        return base_out + self.scale * lora_out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, q_w, k_w, v_w, out_w, gate_w, q_norm_s, k_norm_s, lora_rank=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        
        # Projections with LoRA
        self.q_proj = LoRALinear(q_w, rank=lora_rank)
        self.k_proj = LoRALinear(k_w, rank=lora_rank)
        self.v_proj = LoRALinear(v_w, rank=lora_rank)
        self.out_proj = LoRALinear(out_w, rank=lora_rank)
        self.gate_proj = LoRALinear(gate_w, rank=lora_rank)
        
        self.q_norm = ZCRMSNorm(self.head_dim)
        self.q_norm.scale.data.copy_(q_norm_s)
        self.k_norm = ZCRMSNorm(self.head_dim)
        self.k_norm.scale.data.copy_(k_norm_s)

    def forward(self, x, mask=None, rope=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        gate = torch.sigmoid(self.gate_proj(x))
        out = out * gate
        return self.out_proj(out)


def _sinkhorn(logits, iters=20):
    log_K = logits
    for _ in range(iters):
        log_K = log_K - torch.logsumexp(log_K, dim=-1, keepdim=True)
        log_K = log_K - torch.logsumexp(log_K, dim=-2, keepdim=True)
    return torch.exp(log_K)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, weights, lora_rank=16):
        super().__init__()
        sa_w = weights["self_attn"]
        q_w = sa_w["q_proj"]["kernel"]
        k_w = sa_w["k_proj"]["kernel"]
        v_w = sa_w["v_proj"]["kernel"]
        out_w = sa_w["out_proj"]["kernel"]
        gate_w = sa_w["gate_proj"]["kernel"]
        q_norm_s = sa_w["q_norm"]["scale"]
        k_norm_s = sa_w["k_norm"]["scale"]

        self.pre_attn_norm = ZCRMSNorm(d_model)
        self.pre_attn_norm.scale.data.copy_(weights["ZCRMSNorm_0"]["scale"])

        self.self_attn = MultiHeadAttention(
            d_model, num_heads, num_kv_heads,
            q_w, k_w, v_w, out_w, gate_w, q_norm_s, k_norm_s,
            lora_rank=lora_rank
        )

        self.post_attn_norm = ZCRMSNorm(d_model)
        self.post_attn_norm.scale.data.copy_(weights["post_attn_norm"]["scale"])

        self.attn_gate = nn.Parameter(weights["attn_gate"])

        self.pre_hada_norm = ZCRMSNorm(d_model)
        self.pre_hada_norm.scale.data.copy_(weights["pre_hada_norm"]["scale"])

        self.hadamard_mlp = HadamardMLP(d_model)
        self.hadamard_mlp.d1.data.copy_(weights["hadamard_mlp"]["d1"])
        self.hadamard_mlp.d2.data.copy_(weights["hadamard_mlp"]["d2"])
        self.hadamard_mlp.d3.data.copy_(weights["hadamard_mlp"]["d3"])

    def forward(self, x, mask=None, rope=None):
        skip = x
        normed = self.pre_attn_norm(x)
        attn_out = self.self_attn(normed, mask=mask, rope=rope)
        attn_out = self.post_attn_norm(attn_out)
        x = skip + torch.sigmoid(self.attn_gate) * attn_out

        skip = x
        normed = self.pre_hada_norm(x)
        mlp_out = self.hadamard_mlp(normed)
        return skip + mlp_out


class Needle2Model(nn.Module):
    def __init__(self, ckpt_params, config, lora_rank=16):
        super().__init__()
        self.config = config
        self.d_model = config["d_model"]
        self.num_layers = config["num_layers"]
        self.vocab_size = config["vocab_size"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.mhc_lanes = config.get("mhc_lanes", 4)
        self.embed_scale = 1.0

        # Embedding: [vocab_size, d_model]
        emb_data = torch.from_numpy(np.asarray(ckpt_params["embedding"]["embedding"])).float().to(DEVICE)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embedding.weight.data.copy_(emb_data)
        self.embedding.weight.requires_grad = False

        # Stack weights
        stack = ckpt_params["stack"]
        layers = stack["layers"]["block"]

        # MHC Parameters
        n, L, nC = self.mhc_lanes, self.num_layers, self.mhc_lanes * self.d_model
        lane = torch.eye(n, device=DEVICE)[torch.arange(L, device=DEVICE) % n]
        
        self.register_buffer("mhc_phi_pre", torch.from_numpy(np.asarray(stack["mhc_phi_pre"])).float().to(DEVICE))
        self.register_buffer("mhc_phi_post", torch.from_numpy(np.asarray(stack["mhc_phi_post"])).float().to(DEVICE))
        self.register_buffer("mhc_phi_res", torch.from_numpy(np.asarray(stack["mhc_phi_res"])).float().to(DEVICE))
        self.register_buffer("mhc_b_pre", torch.from_numpy(np.asarray(stack["mhc_b_pre"])).float().to(DEVICE))
        self.register_buffer("mhc_b_post", torch.from_numpy(np.asarray(stack["mhc_b_post"])).float().to(DEVICE))
        self.register_buffer("mhc_b_res", torch.from_numpy(np.asarray(stack["mhc_b_res"])).float().to(DEVICE))
        self.register_buffer("mhc_a_pre", torch.from_numpy(np.asarray(stack["mhc_a_pre"])).float().to(DEVICE))
        self.register_buffer("mhc_a_post", torch.from_numpy(np.asarray(stack["mhc_a_post"])).float().to(DEVICE))
        self.register_buffer("mhc_a_res", torch.from_numpy(np.asarray(stack["mhc_a_res"])).float().to(DEVICE))
        self.register_buffer("pre_off", 8.0 * lane - 4.0)
        self.register_buffer("post_off", -4.0 * (1.0 - lane))

        # Build 27 Transformer Blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            block_weights = {
                "ZCRMSNorm_0": {"scale": torch.from_numpy(np.asarray(layers["ZCRMSNorm_0"]["scale"][layer_idx])).float().to(DEVICE)},
                "attn_gate": torch.from_numpy(np.asarray(layers["attn_gate"][layer_idx])).float().to(DEVICE),
                "post_attn_norm": {"scale": torch.from_numpy(np.asarray(layers["post_attn_norm"]["scale"][layer_idx])).float().to(DEVICE)},
                "pre_hada_norm": {"scale": torch.from_numpy(np.asarray(layers["pre_hada_norm"]["scale"][layer_idx])).float().to(DEVICE)},
                "self_attn": {
                    "q_proj": {"kernel": torch.from_numpy(np.asarray(layers["self_attn"]["q_proj"]["kernel"][layer_idx])).float().to(DEVICE)},
                    "k_proj": {"kernel": torch.from_numpy(np.asarray(layers["self_attn"]["k_proj"]["kernel"][layer_idx])).float().to(DEVICE)},
                    "v_proj": {"kernel": torch.from_numpy(np.asarray(layers["self_attn"]["v_proj"]["kernel"][layer_idx])).float().to(DEVICE)},
                    "out_proj": {"kernel": torch.from_numpy(np.asarray(layers["self_attn"]["out_proj"]["kernel"][layer_idx])).float().to(DEVICE)},
                    "gate_proj": {"kernel": torch.from_numpy(np.asarray(layers["self_attn"]["gate_proj"]["kernel"][layer_idx])).float().to(DEVICE)},
                    "q_norm": {"scale": torch.from_numpy(np.asarray(layers["self_attn"]["q_norm"]["scale"][layer_idx])).float().to(DEVICE)},
                    "k_norm": {"scale": torch.from_numpy(np.asarray(layers["self_attn"]["k_norm"]["scale"][layer_idx])).float().to(DEVICE)},
                },
                "hadamard_mlp": {
                    "d1": torch.from_numpy(np.asarray(layers["hadamard_mlp"]["d1"][layer_idx])).float().to(DEVICE),
                    "d2": torch.from_numpy(np.asarray(layers["hadamard_mlp"]["d2"][layer_idx])).float().to(DEVICE),
                    "d3": torch.from_numpy(np.asarray(layers["hadamard_mlp"]["d3"][layer_idx])).float().to(DEVICE),
                }
            }
            self.blocks.append(TransformerBlock(self.d_model, self.num_heads, self.num_kv_heads, block_weights, lora_rank=lora_rank))

        self.final_norm = ZCRMSNorm(self.d_model)
        self.final_norm.scale.data.copy_(torch.from_numpy(np.asarray(stack["final_norm"]["scale"])).float().to(DEVICE))
        self.final_norm.scale.requires_grad = False

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids) * self.embed_scale
        
        # Precompute RoPE
        cos, sin = precompute_rope_freqs(self.d_model // self.num_heads, T, theta=100000.0, device=input_ids.device)
        rope = (cos, sin)
        
        # Causal mask: upper triangular is -inf
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1)
        causal_mask = causal_mask[None, None, :, :]

        # MHC Stream init: [B, T, n, d_model]
        n = self.mhc_lanes
        x_mhc = x[:, :, None, :].expand(B, T, n, self.d_model).clone()

        for l_idx, block in enumerate(self.blocks):
            nx = _rms_unit(x_mhc.reshape(B, T, n * self.d_model))
            
            # Pre-block mixing
            hpre = torch.sigmoid(
                self.mhc_a_pre[l_idx] * (nx @ self.mhc_phi_pre[l_idx])
                + self.mhc_b_pre[l_idx] + self.pre_off[l_idx]
            )
            u = torch.einsum("btn,btnc->btc", hpre, x_mhc)
            
            # Block forward
            y = block(u, mask=causal_mask, rope=rope) - u
            
            # Post-block mixing
            hpost = 2.0 * torch.sigmoid(
                self.mhc_a_post[l_idx] * (nx @ self.mhc_phi_post[l_idx])
                + self.mhc_b_post[l_idx] + self.post_off[l_idx]
            )
            res = nx @ self.mhc_phi_res[l_idx]
            hres = _sinkhorn(self.mhc_a_res[l_idx] * res.reshape(B, T, n, n) + self.mhc_b_res[l_idx])
            
            x_mhc = torch.einsum("btij,btjc->btic", hres, x_mhc) + hpost[..., None] * y[:, :, None, :]

        # Mean across MHC streams
        x_out = torch.mean(x_mhc, dim=2)
        x_out = self.final_norm(x_out)
        
        # LM Head: x @ embedding.weight.T
        logits = x_out @ self.embedding.weight.T
        return logits
