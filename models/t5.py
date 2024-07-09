import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors
from dataclasses import dataclass

@dataclass
class T5Config:
    architectures: list[str]
    classifier_dropout: float
    d_ff: int
    d_kv: int
    d_model: int
    decoder_start_token_id: int
    dense_act_fn: str
    dropout_rate: float
    eos_token_id: int
    feed_forward_proj: str
    initializer_factor: float
    is_encoder_decoder: bool
    is_gated_act: bool
    layer_norm_epsilon: float
    model_type: str
    num_decoder_layers: int
    num_heads: int
    num_layers: int
    output_past: bool
    pad_token_id: int
    relative_attention_max_distance: int
    relative_attention_num_buckets: int
    tie_word_embeddings: bool
    torch_dtype: str
    transformers_version: str
    use_cache: bool
    vocab_size: int


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = GELU()

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        assert config.is_gated_act
        self.DenseReluDense = T5DenseGatedActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.head_dim = config.d_kv
        self.num_heads = config.num_heads
        self.dropout = config.dropout_rate
        inner_dim = self.num_heads * self.head_dim

        self.q = nn.Linear(config.d_model, inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, config.d_model, bias=False)

    def forward(self, hidden_states, position_bias):
        B, T, C = hidden_states.shape

        q = self.q(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(3, 2)) + position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (B, n_heads, T, T)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)  # (B, n_heads, T, T)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, dim)
        attn_output = self.o(attn_output)
        return attn_output


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.SelfAttention = T5Attention(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, position_bias):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, position_bias)
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states


class T5Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config))
        self.layer.append(T5LayerFF(config))

    def clamp(self, hidden_states):
        if hidden_states.dtype == torch.float16:
            f16_max = torch.finfo(torch.float16).max
            clamp_value = torch.where(torch.isinf(hidden_states).any(), f16_max - 1000, f16_max)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states

    def forward(self, hidden_states, position_bias):
        hidden_states = self.layer[0](hidden_states, position_bias)
        hidden_states = self.clamp(hidden_states)
        hidden_states = self.layer[1](hidden_states)
        hidden_states = self.clamp(hidden_states)
        return hidden_states


class T5Stack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, config.num_heads)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.block = nn.ModuleList([T5Block(config) for _ in range(config.num_layers)])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def relative_position_bucket(self, relative_position):
        num_buckets = self.config.relative_attention_num_buckets // 2
        max_distance = self.config.relative_attention_max_distance
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        # half of the buckets are for exact increments in positions
        # the other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        max_exact = num_buckets // 2
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        return relative_buckets + torch.where(relative_position < max_exact, relative_position, relative_position_if_large)

    def forward(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.dropout(inputs_embeds)

        positions = torch.arange(inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device)
        relative_position = positions[None, :] - positions[:, None]  # shape (T, T)
        relative_position_bucket = self.relative_position_bucket(relative_position)
        position_bias = self.relative_attention_bias(relative_position_bucket)  # (T, T, H)
        position_bias = position_bias.permute(2, 0, 1)[None]  # (1, H, T, T)

        for layer_module in self.block:
            hidden_states = layer_module(hidden_states, position_bias)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class T5EncoderModel(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.encoder = T5Stack(config)

    def forward(self, input_ids):
        return self.encoder(input_ids)

    @classmethod
    def from_pretrained(cls, path):
        with open(path / 'config.json') as f:
            config = T5Config(**json.load(f))
        model = cls(config).eval()
        state_dict = {}
        with safetensors.safe_open(path / 'model-00001-of-00002.safetensors', framework="pt", device='cpu') as f:
            state_dict.update({k: f.get_tensor(k) for k in f.keys()})
        with safetensors.safe_open(path / 'model-00002-of-00002.safetensors', framework="pt", device='cpu') as f:
            state_dict.update({k: f.get_tensor(k) for k in f.keys()})
        state_dict['encoder.embed_tokens.weight'] = state_dict.pop('shared.weight')
        state_dict['encoder.relative_attention_bias.weight'] = state_dict.pop('encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight')
        model.load_state_dict(state_dict)
        return model


if __name__ == '__main__':
    from pathlib import Path
    from transformers import T5Tokenizer
    from transformers import T5EncoderModel as T5EncoderModelHF
    ROOT_DIR = Path('/home/batman/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671')
    prompt = "A cat riding a horse"
    device = torch.device('cpu')

    tokenizer = T5Tokenizer.from_pretrained(ROOT_DIR / 'tokenizer_3')
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    my_t5 = T5EncoderModel.from_pretrained(ROOT_DIR / 'text_encoder_3').to(device)
    gt_t5 = T5EncoderModelHF.from_pretrained(ROOT_DIR / 'text_encoder_3').to(device)
    my_output = my_t5(text_inputs.input_ids.to(device))
    gt_output = gt_t5(text_inputs.input_ids.to(device))
    torch.testing.assert_close(gt_output.last_hidden_state.cpu(), my_output.cpu(), rtol=1e-4, atol=1e-4)

    print("All tests passed!")
