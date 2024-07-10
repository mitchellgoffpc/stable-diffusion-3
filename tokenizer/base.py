import os
import json
import numpy as np

class PreTrainedTokenizer:
    vocab_files_names: dict[str, str] = {}

    def __init__(self, added_tokens_decoder, bos_token=None, eos_token=None, pad_token=None, unk_token=None, **kwargs):
        self._added_tokens_decoder = {int(k): v['content'] for k, v in added_tokens_decoder.items()}
        self._added_tokens_encoder = {k: v for v, k in self._added_tokens_decoder.items()}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

    def __call__(self, text: str, max_length: int) -> np.ndarray:
        tokens = self.tokenize(text)
        ids = [self.convert_token_to_id(x) for x in tokens]

        total_len = len(self.build_inputs_with_special_tokens(ids))
        if total_len > max_length:
            ids = self.truncate_sequences(ids, num_tokens_to_remove=total_len - max_length)
        ids = self.build_inputs_with_special_tokens(ids)
        ids = self.pad(ids, max_length)

        return np.array(ids)[None]

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def truncate_sequences(self, ids: list[int], num_tokens_to_remove: int) -> list[int]:
        if num_tokens_to_remove <= 0:
            return ids
        window_len = min(len(ids), num_tokens_to_remove)
        ids = ids[:-num_tokens_to_remove]
        return ids

    def pad(self, encoded_inputs: list[int], max_length: int) -> list[int]:
        if len(encoded_inputs) != max_length:
            difference = max_length - len(encoded_inputs)
            encoded_inputs = encoded_inputs + [self.pad_token_id] * difference
        return encoded_inputs

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [self.bos_token_id] + token_ids + [self.eos_token_id]

    def convert_token_to_id(self, token: str) -> int:
        if token in self._added_tokens_encoder:
            return self._added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    @property
    def bos_token_id(self) -> int:
        return self.convert_token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.convert_token_to_id(self.eos_token)

    @property
    def pad_token_id(self) -> int:
        return self.convert_token_to_id(self.pad_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        vocab_files = {
            **cls.vocab_files_names,
            "tokenizer_config_file": "tokenizer_config.json",
            "tokenizer_file": "tokenizer.json",
        }

        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            full_file_path = os.path.join(str(pretrained_model_name_or_path), file_path)
            resolved_vocab_files[file_id] = full_file_path if os.path.exists(full_file_path) else None

        with open(resolved_vocab_files["tokenizer_config_file"], encoding="utf-8") as f:
            init_kwargs = resolved_vocab_files | json.load(f)

        return cls(**init_kwargs)
