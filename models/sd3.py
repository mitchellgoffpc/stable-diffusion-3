import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import safetensors
from dataclasses import dataclass

@dataclass
class SD3Transformer2DModelConfig:
    _class_name: str
    _diffusers_version: str
    sample_size: int
    patch_size: int
    in_channels: int
    num_layers: int
    attention_head_dim: int
    num_attention_heads: int
    joint_attention_dim: int
    caption_projection_dim: int
    pooled_projection_dim: int
    out_channels: int
    pos_embed_max_size: int


class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, conditioning_embedding):
        emb = self.linear(F.silu(conditioning_embedding))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, conditioning_embedding):
        emb = self.linear(F.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class LinearGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate='tanh')
        return hidden_states

class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(LinearGELU(dim, dim * 4))
        self.net.append(nn.Identity())
        self.net.append(nn.Linear(dim * 4, dim_out))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class PatchEmbed(nn.Module):
    def __init__(self, height: int, width: int, patch_size: int, in_channels: int, embed_dim: int, pos_embed_max_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.height, self.width = height // patch_size, width // patch_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size)
        pos_embed = PatchEmbed.get_2d_sincos_pos_embed(embed_dim, grid_size=pos_embed_max_size, base_size=self.height)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float()[None], persistent=True)

    @staticmethod
    def get_1d_sincos_pos_embed(embed_dim, pos):
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)
        return emb

    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim, grid_size, base_size):
        grid_h = np.arange(grid_size, dtype=np.float32) / (grid_size / base_size)
        grid_w = np.arange(grid_size, dtype=np.float32) / (grid_size / base_size)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        emb_h = PatchEmbed.get_1d_sincos_pos_embed(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = PatchEmbed.get_1d_sincos_pos_embed(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def forward(self, latent):
        height, width = latent.shape[-2:]
        height = height // self.patch_size
        width = width // self.patch_size
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2

        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

        return (latent + spatial_pos_embed).to(latent.dtype)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.linear_2(F.silu(sample))
        return sample

class TextEmbedding(nn.Module):
    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.linear_2(F.silu(hidden_states))
        return hidden_states

class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TextEmbedding(pooled_projection_dim, embedding_dim)

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class Attention(nn.Module):
    def __init__(self, query_dim: int, num_heads: int, context_pre_only: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(query_dim, query_dim)
        self.to_v = nn.Linear(query_dim, query_dim)

        self.add_k_proj = nn.Linear(query_dim, query_dim)
        self.add_v_proj = nn.Linear(query_dim, query_dim)
        self.add_q_proj = nn.Linear(query_dim, query_dim)

        self.to_out = nn.ModuleList([nn.Linear(query_dim, query_dim)])
        self.to_add_out = None
        if not context_pre_only:
            self.to_add_out = nn.Linear(query_dim, query_dim)

    def forward(self, hidden_states, encoder_hidden_states):
        B, TH, _ = hidden_states.shape
        _, TE, _ = encoder_hidden_states.shape
        T = TH + TE

        q = torch.cat([self.to_q(hidden_states), self.add_q_proj(encoder_hidden_states)], dim=1).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = torch.cat([self.to_k(hidden_states), self.add_k_proj(encoder_hidden_states)], dim=1).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = torch.cat([self.to_v(hidden_states), self.add_v_proj(encoder_hidden_states)], dim=1).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        hidden_states, encoder_hidden_states = hidden_states[:, :TH], hidden_states[:, TH:]

        hidden_states = self.to_out[0](hidden_states)
        if self.to_add_out is not None:
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states

class JointTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, context_pre_only=False):
        super().__init__()

        self.context_pre_only = context_pre_only
        self.norm1 = AdaLayerNormZero(dim)
        self.attn = Attention(query_dim=dim, num_heads=num_attention_heads, context_pre_only=context_pre_only)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim)

        if context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(dim, dim)
        else:
            self.norm1_context = AdaLayerNormZero(dim)
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim)

    def forward(self, hidden_states, encoder_hidden_states, temb):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, temb)

        attn_output, context_attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states)
        hidden_states = hidden_states + gate_msa[:, None] * attn_output
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp[:, None] * self.ff(norm_hidden_states)

        if not self.context_pre_only:
            encoder_hidden_states = encoder_hidden_states + c_gate_msa[:, None] * context_attn_output
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp[:, None] * self.ff_context(norm_encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class SD3Transformer2DModel(nn.Module):
    def __init__(self, config: SD3Transformer2DModelConfig):
        super().__init__()
        self.config = config
        self.inner_dim = config.num_attention_heads * config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=config.sample_size,
            width=config.sample_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=config.pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(embedding_dim=self.inner_dim, pooled_projection_dim=config.pooled_projection_dim)
        self.context_embedder = nn.Linear(config.joint_attention_dim, config.caption_projection_dim)

        self.transformer_blocks = nn.ModuleList([
            JointTransformerBlock(self.inner_dim, num_attention_heads=config.num_attention_heads, context_pre_only=i == config.num_layers - 1)
            for i in range(config.num_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, config.patch_size * config.patch_size * config.out_channels)

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep):
        height, width = hidden_states.shape[-2:]
        patch_size = self.config.patch_size

        hidden_states = self.pos_embed(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], height // patch_size, width // patch_size, patch_size, patch_size, self.config.out_channels)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], self.config.out_channels, height, width)
        return hidden_states

    @classmethod
    def from_pretrained(cls, path):
        with open(path / 'config.json') as f:
            config = SD3Transformer2DModelConfig(**json.load(f))
        model = cls(config).eval()
        with safetensors.safe_open(path / 'diffusion_pytorch_model.safetensors', framework="pt", device='cpu') as f:
            model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
        return model


if __name__ == '__main__':
    from pathlib import Path
    from diffusers.models.transformers import SD3Transformer2DModel as SD3Transformer2DModelHF
    ROOT_DIR = Path('/home/batman/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671')

    device = torch.device('cpu')
    hidden_states = torch.randn(2, 16, 96, 128).to(device)
    timestep = torch.randn(2).to(device)
    encoder_hidden_states = torch.randn(2, 333, 4096).to(device)
    pooled_projections = torch.randn(2, 2048).to(device)

    my_sd3 = SD3Transformer2DModel.from_pretrained(ROOT_DIR / 'transformer').to(device)
    gt_sd3 = SD3Transformer2DModelHF.from_pretrained(ROOT_DIR / 'transformer').to(device)
    my_output = my_sd3(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, pooled_projections=pooled_projections)
    gt_output = gt_sd3(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, pooled_projections=pooled_projections)
    torch.testing.assert_close(gt_output.sample, my_output, rtol=1e-4, atol=1e-4)

    print("All tests passed!")
