import torch
import numpy as np
from pathlib import Path
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A man in a space suit on the Paris metro at rush hour, waiting to get off at his stop."

prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
    prompt=prompt,
    prompt_2=None,
    prompt_3=None,
    negative_prompt="",
    device=pipe._execution_device,
)

np.savez(
    Path(__file__).parent / 'prompt_embeds.npz',
    prompt_embeds=prompt_embeds.detach().cpu().numpy(),
    pooled_prompt_embeds=pooled_prompt_embeds.detach().cpu().numpy(),
    negative_prompt_embeds=negative_prompt_embeds.detach().cpu().numpy(),
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.detach().cpu().numpy()
)


# image = pipe(
#     prompt,
#     # negative_prompt="",
#     num_inference_steps=28,
#     guidance_scale=1,
# ).images[0]
# image.save(Path(__file__).parent / 'output2_cfg1.jpg')

# image = pipe(
#     prompt,
#     # negative_prompt="",
#     num_inference_steps=28,
#     guidance_scale=1.25,
# ).images[0]
# image.save(Path(__file__).parent / 'output2_cfg125.jpg')

# image = pipe(
#     prompt,
#     # negative_prompt="",
#     num_inference_steps=28,
#     guidance_scale=1.5,
# ).images[0]
# image.save(Path(__file__).parent / 'output2_cfg150.jpg')

# image = pipe(
#     prompt,
#     # negative_prompt="",
#     num_inference_steps=28,
#     guidance_scale=1.75,
# ).images[0]
# image.save(Path(__file__).parent / 'output2_cfg175.jpg')

# image = pipe(
#     prompt,
#     # negative_prompt="",
#     num_inference_steps=28,
#     guidance_scale=2,
# ).images[0]
# image.save(Path(__file__).parent / 'output2_cfg2.jpg')