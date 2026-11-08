"""OmniGen2 adapter for the unified UMM protocol."""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock

import torch

from baselines.umm_framework import (
    BackendCapabilities,
    StatelessMinimalBackend,
    _normalise_previous_images,
    seeded_generator,
)
from baselines.umm_model_identity import model_artifact_identity


class OmniGen2Backend(StatelessMinimalBackend):
    name = "omnigen2"
    capabilities = BackendCapabilities(
        name=name,
        condition_images_consumed=True,
        previous_image_consumed=True,
        multiple_condition_images_consumed=True,
        requested_shape_consumed=True,
        requested_image_steps_consumed=True,
        requested_seed_consumed=True,
        reviewer_supported=True,
        planner_transport="concatenated system and user prompt",
        planner_context_limit=128000,
        image_config={
            "sampler": "official OmniGen2 diffusion pipeline",
            "num_inference_steps": "image_steps",
            "text_guidance_scale": 4.0,
            "image_guidance_scale": {
                "without_previous_image": 1.0,
                "with_previous_image": 2.0,
            },
            "cfg_range": [0.0, 1.0],
            "generation_thread_safety": "serialized adapter calls",
        },
    )

    def __init__(self, repo_path: str, model_path: str, device: str):
        repo = Path(repo_path).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from omnigen2.models.transformers.transformer_omnigen2 import (
            OmniGen2Transformer2DModel,
        )
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import (
            OmniGen2ChatPipeline,
        )

        self._device = torch.device(device)
        self._generation_lock = Lock()
        self._pipeline = OmniGen2ChatPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self._pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self._pipeline = self._pipeline.to(self._device)
        self.model_identity = model_artifact_identity(model_path)

    def planner_token_counts(self, system_text, user_text, response):
        """Planner prompt/response token counts for this backend."""
        del system_text
        tokenizer = self._pipeline.processor.tokenizer
        prompt = self._pipeline._apply_chat_template(user_text, None)
        return (len(tokenizer(prompt).input_ids),
                len(tokenizer(response).input_ids))

    def generate_text(self, prompt, previous_image, max_tokens, temperature):
        input_images = _normalise_previous_images(previous_image, None) or None
        prompt = self._pipeline._apply_chat_template(prompt, input_images)
        inputs = self._pipeline.prepare_inputs_for_text_generation(
            prompt, input_images, self._pipeline.mllm.device
        )
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "stop_strings": ["<|im_end|>", "<|img|>", "<|endoftext|>"],
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        generated_ids = self._pipeline.mllm.generate(
            **inputs,
            tokenizer=self._pipeline.processor.tokenizer,
            **generation_kwargs,
        )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self._pipeline.processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0].replace("<|im_end|>", "").strip()

    @torch.inference_mode()
    def generate_image(
        self,
        content,
        previous_image,
        seed,
        image_shape,
        image_steps,
        previous_images=None,
    ):
        height, width = image_shape
        input_images = _normalise_previous_images(previous_image, previous_images)
        input_images = input_images or None
        prompt = self._pipeline._apply_chat_template(content, input_images)
        generator = seeded_generator(self._device, seed)
        with self._generation_lock:
            images = self._pipeline.generate_image(
                prompt=prompt,
                input_images=input_images,
                width=width,
                height=height,
                max_pixels=height * width,
                max_input_image_side_length=max(height, width),
                num_inference_steps=image_steps,
                text_guidance_scale=4.0,
                image_guidance_scale=2.0 if input_images else 1.0,
                cfg_range=(0.0, 1.0),
                generator=generator,
                output_type="pil",
                return_dict=False,
            )
        if not images:
            raise RuntimeError("OmniGen2 returned no image")
        return images[0]


def load_omnigen2_backend(args):
    return OmniGen2Backend(args.repo_path, args.model_path, args.device)
