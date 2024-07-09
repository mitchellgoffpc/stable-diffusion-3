import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors
from dataclasses import dataclass

ACT2FN = {
    'gelu': F.gelu,
    'quick_gelu': lambda x: x * torch.sigmoid(1.702 * x)}

@dataclass
class CLIPTextConfig:
    architectures: list[str]
    attention_dropout: float
    bos_token_id: int
    dropout: float
    eos_token_id: int
    hidden_act: str
    hidden_size: int
    initializer_factor: float
    initializer_range: float
    intermediate_size: int
    layer_norm_eps: float
    max_position_embeddings: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    pad_token_id: int
    projection_dim: int
    torch_dtype: str
    transformers_version: str
    vocab_size: int

@dataclass
class CLIPTextModelOutput:
    text_embeds: torch.tensor
    hidden_states: torch.tensor


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids):
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)

class CLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.config.attention_dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)

        return x, hidden_states

class CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        assert config.eos_token_id == 2
        self.config = config
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        x, hidden_states = self.encoder(x)
        x = self.final_layer_norm(x)

        pooled_output = x[
            torch.arange(x.shape[0], device=x.device),
            input_ids.to(dtype=torch.int, device=x.device).argmax(dim=-1)]

        return pooled_output, hidden_states

class CLIPTextModelWithProjection(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.text_model = CLIPTextTransformer(config)
        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

    def forward(self, input_ids):
        pooled_output, hidden_states = self.text_model(input_ids)
        text_embeds = self.text_projection(pooled_output)
        return CLIPTextModelOutput(text_embeds=text_embeds, hidden_states=hidden_states)

    @classmethod
    def from_pretrained(cls, path):
        with open(path / 'config.json') as f:
            config = CLIPTextConfig(**json.load(f))
        model = cls(config).eval()
        with safetensors.safe_open(path / 'model.safetensors', framework="pt", device='cpu') as f:
            model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
        return model


if __name__ == '__main__':
    from pathlib import Path
    from transformers import CLIPTokenizer
    from transformers import CLIPTextModelWithProjection as CLIPTextModelWithProjectionHF
    ROOT_DIR = Path('/home/batman/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671')
    prompt = "A cat riding a horse"

    for tokenizer_name, text_encoder_name in [('tokenizer', 'text_encoder'), ('tokenizer_2', 'text_encoder_2')]:
        tokenizer = CLIPTokenizer.from_pretrained(ROOT_DIR / tokenizer_name)
        text_input_ids = tokenizer(
            [prompt],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        gt_clip = CLIPTextModelWithProjectionHF.from_pretrained(ROOT_DIR / text_encoder_name)
        my_clip = CLIPTextModelWithProjection.from_pretrained(ROOT_DIR / text_encoder_name)
        gt_output = gt_clip(text_input_ids, output_hidden_states=True)
        my_output = my_clip(text_input_ids)
        torch.testing.assert_close(gt_output.text_embeds, my_output.text_embeds, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(gt_output.hidden_states, my_output.hidden_states, rtol=1e-4, atol=1e-4)

    print("All tests passed!")