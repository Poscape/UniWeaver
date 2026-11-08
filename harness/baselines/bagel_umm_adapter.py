"""BAGEL adapter for the shared UMM minimal and UniWeaver runners."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy

import torch
from PIL import Image

from bagel_text_backend import BagelTextBackend
from baselines.umm_framework import (
    STEP_REQUEST_PROMPT,
    STEP_SYSTEM_PROMPT,
    BackendCapabilities,
    UMMRunConfig,
    _normalise_previous_images,
    parse_step_text,
)
from core.inferencer_adapter import UniWeaverInterleaveInferencer
from core.production import image_generation_config, image_seed_for
from data.data_utils import pil_img2rgb


def _inference_context():
    enabled = bool(torch.cuda.is_available())
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(
            "cuda", enabled=enabled, dtype=torch.bfloat16
        )
    return nullcontext()


class BagelBackend:
    """Expose BAGEL native primitives through the common UMM adapter contract."""

    name = "bagel"
    capabilities = BackendCapabilities(
        name=name,
        condition_images_consumed=True,
        previous_image_consumed=True,
        multiple_condition_images_consumed=True,
        requested_shape_consumed=True,
        requested_image_steps_consumed=True,
        requested_seed_consumed=True,
        reviewer_supported=True,
        native_model_backend=True,
        planner_transport="load_multimodal_model-compatible BAGEL text protocol",
        planner_context_limit=32768,
        image_config=image_generation_config(),
    )

    def __init__(self, bundle):
        self.model_identity = bundle.model_identity
        self.sft_checkpoint_training_identity = bundle.sft_metadata
        self._text = BagelTextBackend(
            bundle.model,
            bundle.tokenizer,
        )
        inferencer_kwargs = {
            "model": bundle.model,
            "vae_model": bundle.vae_model,
            "tokenizer": bundle.tokenizer,
            "new_token_ids": bundle.new_token_ids,
            "vit_transform": bundle.vit_transform,
            "vae_transform": bundle.vae_transform,
        }
        self._uniweaver = UniWeaverInterleaveInferencer(**inferencer_kwargs)
        self._minimal_context = None

    def generate_planner_text(
        self, *, user_text, system_text, max_tokens, do_sample, temperature
    ):
        return self._text.send(
            system_text,
            user_text,
            max_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    def planner_token_counts(self, system_text, user_text, response):
        return (
            self._text.input_token_count(system_text, user_text),
            self._text.output_token_count(response),
        )

    def generate_text(self, prompt, previous_image, max_tokens, temperature):
        if previous_image is not None:
            raise ValueError("BAGEL planner text does not accept a previous image")
        return self.generate_planner_text(
            user_text=prompt,
            system_text="",
            max_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )

    @torch.inference_mode()
    def _generate_image_from_inputs(
        self, input_lists, seed, image_shape, image_steps
    ):
        result = self._uniweaver.interleave_inference(
            input_lists,
            think=False,
            understanding_output=False,
            do_sample=False,
            text_temperature=0.0,
            image_shapes=image_shape,
            image_seed=seed,
            **{
                **image_generation_config(),
                "num_timesteps": image_steps,
            },
        )
        generated = [item for item in result if isinstance(item, Image.Image)]
        if not generated:
            raise RuntimeError("BAGEL returned no decoded image")
        return generated[-1]

    def generate_uniweaver_image(
        self, *, input_lists, seed, image_shape, image_steps
    ):
        """Use the native UniWeaver string/image input sequence unchanged."""
        return self._generate_image_from_inputs(
            input_lists, seed, image_shape, image_steps)

    def generate_image(
        self,
        content,
        previous_image,
        seed,
        image_shape,
        image_steps,
        previous_images=None,
    ):
        images = _normalise_previous_images(previous_image, previous_images)
        return self._generate_image_from_inputs(
            [*images, content], seed, image_shape, image_steps)

    def review_image(self, *, image, prompt, condition_images, max_tokens):
        inputs = []
        for key, reference in condition_images.items():
            inputs.extend((f"[Reference: {key}]", reference))
        if inputs:
            inputs.append("Above are visual references used by the generated Step.")
        inputs.extend((image, prompt))
        output = self._uniweaver.interleave_inference(
            inputs,
            # BAGEL has a think mode and the other backends have no equivalent
            # prompt, so every Plan4 reviewer stays on one forward pass.
            think=False,
            understanding_output=True,
            max_think_token_n=max_tokens,
            do_sample=False,
            text_temperature=0.3,
        )
        return "\n".join(value for value in output if isinstance(value, str)).strip()

    def begin_minimal_task(self, task_prompt: str):
        context = self._uniweaver.init_gen_context()
        context = self._uniweaver.update_context_text(STEP_SYSTEM_PROMPT, context)
        context = self._uniweaver.update_context_text(task_prompt, context)
        self._minimal_context = self._uniweaver.update_context_text(
            STEP_REQUEST_PROMPT, context
        )

    def generate_minimal_step(
        self,
        task_prompt: str,
        history: list[str],
        previous_image: Image.Image | None,
        step_index: int,
        config: UMMRunConfig,
    ):
        del task_prompt, history, previous_image
        if self._minimal_context is None:
            raise RuntimeError("minimal task context was not initialized")
        context = self._minimal_context
        cfg_text_context = deepcopy(context)
        cfg_image_context = deepcopy(context)
        image_config = image_generation_config()
        image_config["num_timesteps"] = config.image_steps
        with torch.no_grad(), _inference_context():
            raw_text = self._uniweaver.gen_text(
                context,
                max_length=config.text_max_tokens,
                do_sample=config.do_sample,
                temperature=config.text_temperature,
            )
            self.last_minimal_step_attempt = {"raw_text": raw_text}
            content, done = parse_step_text(raw_text)
            if not content:
                # Nothing to illustrate or show a reader: stop without a step.
                self._minimal_context = None
                return raw_text, content, done, None, None
            context = self._uniweaver.update_context_text(content, context)
            image_seed = image_seed_for(config.seed, "step", step_index)
            image = self._uniweaver.gen_image(
                config.image_shape,
                context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_image_context,
                image_seed=image_seed,
                **image_config,
            )
            if image is None:
                raise RuntimeError(
                    f"image generation returned no image for step {step_index}"
                )
            if not done:
                prepared_image = self._uniweaver.vae_transform.resize_transform(
                    pil_img2rgb(image)
                )
                context = self._uniweaver.update_context_image(
                    prepared_image, context, vae=True
                )
                context = self._uniweaver.update_context_text(
                    STEP_REQUEST_PROMPT, context
                )
        self._minimal_context = None if done else context
        return raw_text, content, done, image, image_seed


def load_bagel_backend(args):
    from bagel_model_loader import load_multimodal_model

    bundle = load_multimodal_model(
        model_path=args.model_path,
        mode=args.bagel_mode,
        offload_folder=args.offload_folder,
        sft_checkpoint=args.sft_checkpoint,
    )
    return BagelBackend(bundle)
