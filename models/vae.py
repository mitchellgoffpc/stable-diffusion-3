import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors
from typing import Optional
from dataclasses import dataclass

@dataclass
class AutoencoderKLConfig:
    _class_name: str
    _diffusers_version: str
    in_channels: int
    out_channels: int
    down_block_types: tuple[str]
    up_block_types: tuple[str]
    block_out_channels: tuple[int]
    layers_per_block: int
    act_fn: str
    latent_channels: int
    norm_num_groups: int
    sample_size: int
    scaling_factor: float
    shift_factor: Optional[float]
    latents_mean: Optional[tuple[float]]
    latents_std: Optional[tuple[float]]
    force_upcast: float
    use_quant_conv: bool
    use_post_quant_conv: bool


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2)

    def forward(self, hidden_states):
        hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states

class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, query_dim: int, heads: int, dim_head: int, norm_num_groups: int):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.num_heads = heads
        self.head_dim = dim_head

        self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=1e-6, affine=True)
        self.to_q = nn.Linear(query_dim, self.inner_dim)
        self.to_k = nn.Linear(query_dim, self.inner_dim)
        self.to_v = nn.Linear(query_dim, self.inner_dim)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim)])

    def forward(self, input_tensor):
        B, C, H, W = input_tensor.shape

        hidden_states = input_tensor.view(B, C, H * W)
        hidden_states = self.group_norm(hidden_states).transpose(1, 2)

        q = self.to_q(hidden_states).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(hidden_states).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(hidden_states).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(B, H * W, self.inner_dim)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(B, C, H, W)

        return input_tensor + hidden_states


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input_tensor):
        hidden_states = self.norm1(input_tensor)
        hidden_states = self.conv1(F.silu(hidden_states))
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.conv2(F.silu(hidden_states))\

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        return input_tensor + hidden_states


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, resnet_groups: int, add_downsample: bool):
        super().__init__()

        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            self.resnets.append(ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, groups=resnet_groups))

        self.downsamplers = []
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
        return hidden_states

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, resnet_groups: int, add_upsample: bool):
        super().__init__()

        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            self.resnets.append(ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, groups=resnet_groups))

        self.upsamplers = []
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
        return hidden_states

class UNetMidBlock(nn.Module):
    def __init__(self, in_channels: int, resnet_groups: int, attention_head_dim: int):
        super().__init__()
        self.resnets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        self.resnets.append(ResnetBlock2D(in_channels, in_channels, groups=resnet_groups))
        self.attentions.append(Attention(in_channels, heads=in_channels // attention_head_dim, dim_head=attention_head_dim, norm_num_groups=resnet_groups))
        self.resnets.append(ResnetBlock2D(in_channels, in_channels, groups=resnet_groups))

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config: AutoencoderKLConfig):
        super().__init__()
        self.conv_in = nn.Conv2d(config.in_channels, config.block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([])

        in_channels = config.block_out_channels[0]
        for i, out_channels in enumerate(config.block_out_channels):
            self.down_blocks.append(UNetDownBlock(in_channels, out_channels, num_layers=config.layers_per_block, add_downsample=i < len(config.block_out_channels) - 1, resnet_groups=config.norm_num_groups))
            in_channels = out_channels

        self.mid_block = UNetMidBlock(config.block_out_channels[-1], attention_head_dim=config.block_out_channels[-1], resnet_groups=config.norm_num_groups)
        self.conv_norm_out = nn.GroupNorm(num_channels=config.block_out_channels[-1], num_groups=config.norm_num_groups, eps=1e-6)
        self.conv_out = nn.Conv2d(config.block_out_channels[-1], 2 * config.latent_channels, 3, padding=1)

    def forward(self, sample):
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_out(F.silu(sample))
        return sample


class Decoder(nn.Module):
    def __init__(self, config: AutoencoderKLConfig):
        super().__init__()
        self.conv_in = nn.Conv2d(config.latent_channels, config.block_out_channels[-1], kernel_size=3, padding=1)
        self.mid_block = UNetMidBlock(config.block_out_channels[-1], attention_head_dim=config.block_out_channels[-1], resnet_groups=config.norm_num_groups)
        self.up_blocks = nn.ModuleList([])

        in_channels = config.block_out_channels[-1]
        for i, out_channels in enumerate(reversed(config.block_out_channels)):
            self.up_blocks.append(UNetUpBlock(in_channels, out_channels, num_layers=config.layers_per_block + 1, add_upsample=i < len(config.block_out_channels) - 1, resnet_groups=config.norm_num_groups))
            in_channels = out_channels

        self.conv_norm_out = nn.GroupNorm(num_channels=config.block_out_channels[0], num_groups=config.norm_num_groups, eps=1e-6)
        self.conv_out = nn.Conv2d(config.block_out_channels[0], config.out_channels, 3, padding=1)

    def forward(self, sample):
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)
        for up_block in self.up_blocks:
            sample = up_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_out(F.silu(sample))
        return sample


class AutoencoderKL(nn.Module):
    def __init__(self, config: AutoencoderKLConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean, std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior: bool = False):
        mean, std = self.encode(x)
        if sample_posterior:
            z = mean + std * torch.randn(std.shape, device=std.device, dtype=std.dtype)
        else:
            z = mean
        return self.decode(z)

    @classmethod
    def from_pretrained(cls, path):
        with open(path / 'config.json') as f:
            config = AutoencoderKLConfig(**json.load(f))
        model = cls(config).eval()
        with safetensors.safe_open(path / 'diffusion_pytorch_model.safetensors', framework="pt", device='cpu') as f:
            model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
        return model


if __name__ == '__main__':
    from pathlib import Path
    from diffusers import AutoencoderKL as AutoencoderKLHF
    ROOT_DIR = Path('/home/batman/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671')

    device = torch.device('cpu')
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    gt_vae = AutoencoderKLHF.from_pretrained(ROOT_DIR / 'vae').to(device)
    my_vae = AutoencoderKL.from_pretrained(ROOT_DIR / 'vae').to(device)
    gt_output = gt_vae(dummy_input)
    my_output = my_vae(dummy_input)
    torch.testing.assert_close(gt_output.sample, my_output, rtol=1e-4, atol=1e-4)

    print("All tests passed!")
