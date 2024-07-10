import sentencepiece as spm
from tokenizer.base import PreTrainedTokenizer

class T5Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "spiece.model"}

    def __init__(self, vocab_file, **kwargs):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        super().__init__(**kwargs)

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return token_ids + [self.eos_token_id]

    def tokenize(self, text: str) -> list[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)


if __name__ == '__main__':
    import numpy as np
    from pathlib import Path
    from transformers import T5Tokenizer as T5TokenizerHF
    ROOT_DIR = Path('/home/batman/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671')
    prompt_1 = "A cat riding a horse!  🐴"
    prompt_2 = (
        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto "
        "beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. "
        "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. "
        "Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit "
        "esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?"
    )

    my_tokenizer = T5Tokenizer.from_pretrained(ROOT_DIR / 'tokenizer_3')
    gt_tokenizer = T5TokenizerHF.from_pretrained(ROOT_DIR / 'tokenizer_3')

    for prompt in [prompt_1, prompt_2]:
        my_text_input_ids = my_tokenizer(prompt, max_length=256)
        gt_text_input_ids = gt_tokenizer(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        ).input_ids

        np.testing.assert_equal(gt_text_input_ids, my_text_input_ids)

    print("All tests passed!")
