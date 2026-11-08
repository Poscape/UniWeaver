"""ILLUME+ 7B adapter for the unified UMM protocol."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModel, AutoProcessor

from baselines.umm_framework import (
    BackendCapabilities,
    StatelessMinimalBackend,
    _normalise_previous_images,
    pil_lanczos,
)
from baselines.umm_model_identity import model_artifact_identity


def _seeded_rng_context(device):
    """Fork the RNG for this call, CUDA-only, so CPU runs do not poke cuda:0."""
    if device.type == "cuda":
        return torch.random.fork_rng(devices=[device.index or 0])
    return nullcontext()


class IllumePlusBackend(StatelessMinimalBackend):
    name = "illume-plus-7b"
    capabilities = BackendCapabilities(
        name=name,
        condition_images_consumed=True,
        previous_image_consumed=True,
        multiple_condition_images_consumed=False,
        requested_shape_consumed=True,
          requested_image_steps_consumed=True,
          requested_seed_consumed=True,
          reviewer_supported=True,
          planner_transport="official ILLUME+ 7B Hugging Face text head",
          planner_context_limit=None,
        image_config={
            "sampler": "official ILLUME+ dual-token autoregressive generation",
            "detokenizer": "DualViTok plus SDXL diffusion decoder",
            "llm_cfg": 2.0,
            "image_top_k": 2048,
            "image_top_p": 1.0,
        },
    )

    def __init__(
        self,
        repo_path: str,
        model_path: str,
        tokenizer_path: str,
        decoder_path: str,
        device: str,
    ):
        repo = Path(repo_path).resolve()
        illume_root = repo / "ILLUME"
        if str(illume_root) not in sys.path:
            sys.path.insert(0, str(illume_root))
        self._device = torch.device(device)
        self._dtype = torch.bfloat16
        self._lock = Lock()
        # Load the processor first: its remote-code entrypoint directly imports
        # sdxl_decoder_pipe.py, ensuring Transformers copies that dependency into
        # the dynamic-module cache before the model's nested DualViTok import.
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(self._device)
        dualvitok = AutoModel.from_pretrained(
            tokenizer_path,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).eval().to(self._device)
        self._processor.set_vision_tokenizer(dualvitok)
        self._processor.load_diffusion_vision_detokenizer(
            decoder_path, device=self._device
        )
        self.model_identity = model_artifact_identity(
            model_path,
            extra_artifacts={
                "dualvitok": tokenizer_path,
                "diffusion_decoder": decoder_path,
            },
        )

    def _chat_prompt(self, prompt: str) -> str:
        tokenizer = self._processor.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _multimodal_prompt(self, prompt: str) -> str:
        """Wrap an image question in the official app_hf.py conversation template."""
        from illume.constants import DEFAULT_IMAGE_TOKEN
        from illume.conversation import conv_templates

        conversation = conv_templates["qwen2"].copy()
        conversation.append_message(
            conversation.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        )
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt()

    def planner_token_counts(self, system_text, user_text, response):
        """Planner prompt/response token counts for this backend."""
        del system_text
        tokenizer = self._processor.tokenizer
        prompt_ids = tokenizer(self._chat_prompt(user_text)).input_ids
        response_ids = tokenizer(response).input_ids
        return (len(prompt_ids), len(response_ids))

    def generate_text(self, prompt, previous_image, max_tokens, temperature):
        if previous_image is None:
            text = self._chat_prompt(prompt)
            inputs = self._processor(text=text, return_tensors="pt").to(
                self._device
            )
        else:
            text = self._multimodal_prompt(prompt)
            inputs = self._processor(
                text=text, images=[previous_image], return_tensors="pt"
            ).to(self._device)
        generation_kwargs = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_tokens,
            "use_cache": True,
            "pad_token_id": self._processor.tokenizer.pad_token_id,
            "eos_token_id": self._processor.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with self._lock, torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                **generation_kwargs,
            )
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        return self._processor.tokenizer.decode(
            generated, skip_special_tokens=True
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
        sources = _normalise_previous_images(previous_image, previous_images)
        if sources:
            return self._edit_image(
                sources[0], content, seed, image_shape, image_steps
            )
        return self._text_to_image(content, seed, image_shape, image_steps)

    def _edit_image(self, source, content, seed, image_shape, image_steps):
        """Official ILLUME+ editing recipe from the released model card."""
        source = source.convert("RGB")
        original_size = source.size
        prepared = self._processor.transform_image_nearest_resolution_ratio(
            source
        )
        target_resolution = prepared.size[::-1]
        resolution_tag = self._processor.get_resolution_tag_from_resolution(
            target_resolution
        )
        request = self._processor.default_editing_template.format(
            resolution_tag=resolution_tag, content=content
        )
        unconditional = (
            self._processor.default_editing_unconditional_template.format(
                resolution_tag=resolution_tag
            )
        )

        def build(text):
            # The editing template carries one image placeholder, so the
            # processor receives one rendered user turn and one image.
            return self._processor(
                text=self._chat_prompt(text),
                images=[prepared],
                return_tensors="pt",
            ).to(self._device)

        inputs = build(request)
        negative = build(unconditional)
        with self._lock, _seeded_rng_context(self._device):
            if seed is not None:
                torch.manual_seed(int(seed))
                if self._device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
            output_ids = self._model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=2048,
                do_sample=True,
                temperature=1.0,
                top_k=2048,
                top_p=1.0,
                negative_image_prompt_ids=negative.input_ids,
                negative_image_prompt_attention_mask=negative.attention_mask,
                target_image_resolution=target_resolution,
                guidance_scale=1.5,
                image_semantic_temperature=0.8,
                image_semantic_top_k=512,
                image_semantic_top_p=0.8,
                image_pixel_temperature=0.8,
                image_pixel_top_k=512 * 3,
                image_pixel_top_p=0.8,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        full_text = self._processor.tokenizer.decode(
            generated, skip_special_tokens=True
        )
        _text, image_indices, _parts = self._processor.parse_text_image(
            full_text
        )
        if not image_indices or not any(image_indices[0]):
            raise RuntimeError("ILLUME+ editing returned no image tokens")
        images = self._processor.decode_images(
            image_indices,
            target_resolution=target_resolution,
            use_diffusion=True,
            diffusion_cfg_scale=2.0,
            diffusion_num_inference_steps=image_steps,
        )
        if not images:
            raise RuntimeError("ILLUME+ diffusion decoder returned no image")
        edited = self._processor.unpad_and_resize_back(
            images[0], *original_size
        )
        return edited.resize(
            (image_shape[1], image_shape[0]), pil_lanczos()
        )

    def _text_to_image(self, content, seed, image_shape, image_steps):
        height, width = image_shape
        token_height = max(256, height // 2)
        token_width = max(256, width // 2)
        ratio_tag = f"<height_{token_height}><width_{token_width}>"
        request = self._chat_prompt(
            f"Generate an image of {ratio_tag}, the content of image is {content}\n"
        )
        unconditional = self._chat_prompt(
            f"Generate a random image of {ratio_tag}\n"
        )
        inputs = self._processor(
            text=request + ratio_tag, return_tensors="pt"
        ).to(self._device)
        negative = self._processor(
            text=unconditional + ratio_tag, return_tensors="pt"
        ).to(self._device)
        with self._lock, _seeded_rng_context(self._device):
            if seed is not None:
                torch.manual_seed(int(seed))
                if self._device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
            output_ids = self._model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=2048,
                do_sample=True,
                temperature=1.0,
                top_k=2048,
                top_p=1.0,
                negative_image_prompt_ids=negative.input_ids,
                negative_image_prompt_attention_mask=negative.attention_mask,
                target_image_resolution=(token_height, token_width),
                guidance_scale=2.0,
                image_semantic_temperature=1.0,
                image_semantic_top_k=2048,
                image_semantic_top_p=1.0,
                image_pixel_temperature=1.0,
                image_pixel_top_k=6144,
                image_pixel_top_p=1.0,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        full_text = self._processor.tokenizer.decode(
            generated, skip_special_tokens=True
        )
        _text, image_indices, _parts = self._processor.parse_text_image(
            full_text, "<image>"
        )
        if not image_indices or not any(image_indices[0]):
            raise RuntimeError("ILLUME+ returned no image tokens")
        images = self._processor.decode_images(
            image_indices,
            target_resolution=(token_height, token_width),
            use_diffusion=True,
            diffusion_cfg_scale=2.0,
            diffusion_num_inference_steps=image_steps,
        )
        if not images:
            raise RuntimeError("ILLUME+ diffusion decoder returned no image")
        return images[0].resize((width, height), pil_lanczos())


def load_illume_plus_backend(args):
    if not args.vq_path:
        raise SystemExit("--vq-path is required for --backend illume-plus")
    if not args.llm_path:
        raise SystemExit("--llm-path is required for --backend illume-plus")
    return IllumePlusBackend(
        args.repo_path,
        args.model_path,
        args.vq_path,
        args.llm_path,
        args.device,
    )
