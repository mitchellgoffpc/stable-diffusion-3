import json
import numpy as np

class FlowMatchEulerDiscreteScheduler:
    def __init__(self, num_train_timesteps: int, shift: float):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = 1
        self.sigma_min = shift / num_train_timesteps

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps

        sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.timesteps = sigmas * self.num_train_timesteps
        self.sigmas = np.concatenate([sigmas, np.zeros(1)])

    def step(self, model_output, timestep, sample):
        indices = (self.timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        step_index = indices[pos].item()

        sample = sample.float()
        sigma = self.sigmas[step_index]
        denoised = sample - model_output * sigma
        derivative = (sample - denoised) / sigma

        dt = self.sigmas[step_index + 1] - sigma
        prev_sample = sample + derivative * dt
        prev_sample = prev_sample.to(model_output.dtype)

        return prev_sample

    @classmethod
    def from_pretrained(cls, path):
        with open(path / 'scheduler_config.json') as f:
            args = json.load(f)
        args.pop('_class_name')
        args.pop('_diffusers_version')
        return cls(**args)


if __name__ == "__main__":
    import torch
    from pathlib import Path
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler as FlowMatchEulerDiscreteSchedulerHF
    ROOT_DIR = Path('/home/batman/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671')

    torch.manual_seed(0)
    my_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ROOT_DIR / 'scheduler')
    gt_scheduler = FlowMatchEulerDiscreteSchedulerHF.from_pretrained(ROOT_DIR / 'scheduler')

    # Set up the inference steps
    num_inference_steps = 25
    my_scheduler.set_timesteps(num_inference_steps)
    gt_scheduler.set_timesteps(num_inference_steps)

    # Create a random sample and model output
    sample_shape = (1, 3, 64, 64)
    sample = torch.randn(sample_shape)
    model_output = torch.randn(sample_shape)

    for i, t in enumerate(my_scheduler.timesteps):
        my_output = my_scheduler.step(model_output, t, sample)
        gt_output = gt_scheduler.step(model_output, t, sample).prev_sample
        torch.testing.assert_close(my_output, gt_output, atol=1e-4, rtol=1e-4)

        sample = my_output
        model_output = torch.randn(sample_shape)

    print("All tests passed!")