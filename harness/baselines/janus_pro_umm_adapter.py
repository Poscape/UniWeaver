"""Janus-Pro 7B adapter for the unified UMM protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from baselines.umm_framework import (
    BackendCapabilities,
    StatelessMinimalBackend,
    pil_lanczos,
    seeded_generator,
)
from baselines.umm_model_identity import model_artifact_identity


class JanusProBackend(StatelessMinimalBackend):
    name = "janus-pro-7b"
    capabilities = BackendCapabilities(
        name=name,
        condition_images_consumed=False,
        previous_image_consumed=False,
        multiple_condition_images_consumed=False,
        requested_shape_consumed=False,
        requested_image_steps_consumed=False,
        requested_seed_consumed=True,
        reviewer_supported=True,
        planner_transport="official Janus-Pro multimodal language and image heads",
        planner_context_limit=16384,
        image_config={
            "sampler": "official Janus-Pro image-token autoregressive loop",
            "image_token_count": 576,
            "native_shape": [384, 384],
            # Janus-Pro emits a fixed 24x24 VQ grid (576 discrete visual
            # tokens) at 384, and the adapter resizes it to the requested
            # delivery size.
            "autoregressive_visual_tokens": 576,
            "native_grid": [24, 24],
            "delivery_resample": "LANCZOS 384 -> requested shape",
        },
    )

    def __init__(self, repo_path: str, model_path: str, device: str):
        repo = Path(repo_path).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        from transformers import AutoConfig, AutoModelForCausalLM

        del MultiModalityCausalLM
        self._device = torch.device(device)
        config = AutoConfig.from_pretrained(model_path)
        language_config = config.language_config
        language_config._attn_implementation = "eager"
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, language_config=language_config, trust_remote_code=True
        ).to(torch.bfloat16).to(self._device).eval()
        self._processor = VLChatProcessor.from_pretrained(model_path)
        self._tokenizer = self._processor.tokenizer
        self.model_identity = model_artifact_identity(model_path)

    # The JSON-only closing instruction lives in
    # baselines.umm_framework.send_planner_text so every backend receives the
    # same contract.
    def planner_token_counts(self, system_text, user_text, response):
        """Planner prompt/response token counts for this backend."""
        del system_text
        text = self._processor.apply_sft_template_for_multi_turn_prompts(
            conversations=[
                {"role": "User", "content": user_text},
                {"role": "Assistant", "content": ""},
            ],
            sft_format=self._processor.sft_format,
            system_prompt="",
        )
        return (len(self._tokenizer.encode(text)),
                len(self._tokenizer.encode(response)))

    def generate_text(self, prompt, previous_image, max_tokens, temperature):
        if previous_image is None:
            conversation = [
                {"role": "User", "content": prompt},
                {"role": "Assistant", "content": ""},
            ]
            images = []
        else:
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [previous_image],
                },
                {"role": "Assistant", "content": ""},
            ]
            images = [previous_image]
        prepare_inputs = self._processor(
            conversations=conversation, images=images, force_batchify=True
        ).to(self._device, dtype=torch.bfloat16)
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self._model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self._tokenizer.eos_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            use_cache=True,
            temperature=temperature if temperature > 0 else None,
        )
        return self._tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        ).strip()

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
        # Decode at the model's native grid and resize to the delivery geometry.
        del previous_image, previous_images, image_steps
        delivery_shape = (int(image_shape[1]), int(image_shape[0]))
        generator = seeded_generator(self._device, seed, modulus=2**32 - 1)
        return self._generate_image(content, generator, delivery_shape)

    def _generate_image(self, content, generator, delivery_shape):
        width = height = 384
        parallel_size = 1
        image_token_count = 576
        messages = [
            {"role": "User", "content": content},
            {"role": "Assistant", "content": ""},
        ]
        text = self._processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=self._processor.sft_format,
            system_prompt="",
        ) + self._processor.image_start_tag
        input_ids = torch.LongTensor(self._tokenizer.encode(text)).to(self._device)
        tokens = input_ids.unsqueeze(0).repeat(parallel_size * 2, 1)
        tokens[1, 1:-1] = self._processor.pad_id
        inputs_embeds = self._model.language_model.get_input_embeddings()(tokens)
        generated = torch.zeros(
            (parallel_size, image_token_count), dtype=torch.int, device=self._device
        )
        past_key_values = None
        for _index in range(image_token_count):
            outputs = self._model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            logits = self._model.gen_head(outputs.last_hidden_state[:, -1, :])
            logits = logits[1::2] + 5.0 * (logits[0::2] - logits[1::2])
            next_token = torch.multinomial(
                torch.softmax(logits, dim=-1), 1, generator=generator
            )
            generated[:, _index] = next_token.squeeze(-1)
            inputs_embeds = self._model.prepare_gen_img_embeds(
                next_token.repeat(1, 2).reshape(-1)
            ).unsqueeze(1)
        patches = self._model.gen_vision_model.decode_code(
            generated, shape=[parallel_size, 8, width // 16, height // 16]
        )
        array = patches[0].to(torch.float32).cpu().numpy().transpose(1, 2, 0)
        array = np.clip((array + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(array).resize(delivery_shape, pil_lanczos())


def load_janus_pro_backend(args):
    return JanusProBackend(args.repo_path, args.model_path, args.device)
