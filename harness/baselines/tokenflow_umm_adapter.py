"""TokenFlow adapter for the unified UMM protocol."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image

from baselines.umm_framework import (
    BackendCapabilities,
    StatelessMinimalBackend,
    pil_lanczos,
)
from baselines.umm_model_identity import model_artifact_identity

# The understanding tower uses the SigLIP-384 tokenizer distributed with the
# official TokenFlow release.
DEFAULT_UNDERSTANDING_VISION_FILENAME = "tokenflow_siglip_32k.pt"


def _resolve_understanding_vision_path(
    tokenizer_path: str, understanding_vision_path: str | None
) -> Path:
    """Locate the SigLIP-384 VQ tokenizer the understanding tower needs."""
    if understanding_vision_path:
        candidate = Path(understanding_vision_path).expanduser()
    else:
        candidate = (
            Path(tokenizer_path).expanduser().parent
            / DEFAULT_UNDERSTANDING_VISION_FILENAME
        )
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise SystemExit(
            "TokenFlow review needs the released SigLIP-384 VQ tokenizer; "
            f"missing {resolved}. Download {DEFAULT_UNDERSTANDING_VISION_FILENAME}, "
            "place it next to --vq-path, or pass --understanding-vision-path."
        )
    return resolved


def _inner_vq_models(vision_tower) -> list:
    """Collect the VQGAN the two official VQ towers keep inside themselves."""
    models = []
    for attribute in ("vision_tower", "vq_model"):
        value = getattr(vision_tower, attribute, None)
        if isinstance(value, (list, tuple)):
            models.extend(item for item in value if item is not None)
        elif value is not None:
            models.append(value)
    return models


def _materialise_meta_vision_tower(vision_tower, device) -> None:
    """Reload a VQ tower that current Transformers left on the meta device."""
    models = [vision_tower, *_inner_vq_models(vision_tower)]
    if not any(
        parameter.is_meta for model in models for parameter in model.parameters()
    ):
        return
    vision_tower.load_model(device_map=device)


@contextmanager
def _tokenflow_import_compatibility():
    """Scope TokenFlow's compatibility version to its own import."""
    import pkg_resources
    from packaging import version

    distribution = pkg_resources.get_distribution("transformers")
    if version.parse(distribution.version) < version.parse("4.43.4"):
        raise RuntimeError(
            "TokenFlow requires transformers >= 4.43.4; found "
            f"{distribution.version}"
        )
    if distribution.version == "4.43.4":
        yield
        return

    original = pkg_resources.get_distribution

    class CheckedDistribution:
        version = "4.43.4"

        def __getattr__(self, name):
            return getattr(distribution, name)

    def compatible_distribution(name):
        if str(name).lower() == "transformers":
            return CheckedDistribution()
        return original(name)

    pkg_resources.get_distribution = compatible_distribution
    try:
        yield
    finally:
        pkg_resources.get_distribution = original


class TokenFlowBackend(StatelessMinimalBackend):
    name = "tokenflow"
    capabilities = BackendCapabilities(
        name=name,
        condition_images_consumed=False,
        previous_image_consumed=False,
        multiple_condition_images_consumed=False,
        requested_shape_consumed=False,
        requested_image_steps_consumed=False,
        requested_seed_consumed=True,
        reviewer_supported=True,
        planner_transport="official TokenFlow Qwen2.5-14B understanding checkpoint",
        planner_context_limit=32768,
        image_config={
            "sampler": "official TokenFlow 7B autoregressive T2I",
            "cfg": 7.5,
            "top_k": 600,
            "top_p": 0.6,
            "native_resolution": [256, 256],
            # TokenFlow is autoregressive over discrete visual tokens, so
            # ``--image-steps`` does not apply; the VQ token budget is the cost.
            "autoregressive_visual_tokens": 817,
            "native_grid": [16, 16],
            "delivery_resample": "LANCZOS 256 -> requested shape",
        },
    )

    def __init__(
        self,
        repo_path: str,
        model_path: str,
        tokenizer_path: str,
        understanding_path: str,
        device: str,
        understanding_vision_path: str | None = None,
    ):
        repo = Path(repo_path).resolve()
        t2i_root = repo / "t2i"
        i2t_root = repo / "i2t"
        for path in (t2i_root, i2t_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        understanding_vision = _resolve_understanding_vision_path(
            tokenizer_path, understanding_vision_path
        )

        with _tokenflow_import_compatibility():
            import transformers
            from llava.model.language_model.llava_qwen import (
                LlavaQwenConfig,
                LlavaQwenForCausalLM,
            )
            from llava_t2i.dataset.process import crop_and_encode_text_and_img
            from llava_t2i.model.language_model.llava_llama import (
                LlavaLlamaForCausalLM,
            )

        self._device = torch.device(device)
        self._lock = Lock()
        self._crop_and_encode = crop_and_encode_text_and_img
        self._understanding_vision_path = understanding_vision
        self._negative_prompt = (
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, cropped, worst quality, low quality, jpeg artifacts, "
            "signature, watermark, blurry"
        )

        self._image_model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            mm_vision_tower=tokenizer_path,
            torch_dtype=torch.bfloat16,
        ).eval().to(self._device)
        _materialise_meta_vision_tower(
            self._image_model.get_vision_tower(), self._device
        )
        self._image_model.get_vision_tower().to(
            device=self._device, dtype=torch.bfloat16
        )
        self._image_model.config.mm_vision_vq_type = str(
            self._image_model.config.mm_vision_vq_type
        )
        self._image_model.config.use_cache = False
        self._image_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=self._image_model.config.tokenizer_model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self._image_model.reinit_image_token_start_end(self._image_tokenizer)

        self._text_tokenizer = transformers.AutoTokenizer.from_pretrained(
            understanding_path, use_fast=True
        )
        text_config = LlavaQwenConfig.from_pretrained(understanding_path)
        text_config.mm_vision_tower = str(understanding_vision)
        text_config.mm_vision_vq_type = "TOKENFLOW"
        self._text_model = LlavaQwenForCausalLM.from_pretrained(
            understanding_path,
            config=text_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        ).eval().to(self._device)
        self._text_dtype = torch.float16
        self._review_image_processor = (
            self._text_model.get_vision_tower().image_processor
        )
        _materialise_meta_vision_tower(
            self._text_model.get_vision_tower(), self._device
        )
        self._text_model.get_vision_tower().to(
            device=self._device, dtype=self._text_dtype
        )
        from llava.conversation import conv_templates

        self._text_conversation = conv_templates["qwen_2_5"]
        self.model_identity = model_artifact_identity(
            model_path,
            extra_artifacts={
                "tokenizer": tokenizer_path,
                "understanding": understanding_path,
                "understanding_vision": str(understanding_vision),
            },
        )

    def planner_token_counts(self, system_text, user_text, response):
        """Planner prompt/response token counts for this backend."""
        del system_text
        conversation = self._text_conversation.copy()
        conversation.append_message(conversation.roles[0], user_text)
        conversation.append_message(conversation.roles[1], None)
        prompt_ids = self._text_tokenizer(
            conversation.get_prompt(), return_tensors="pt").input_ids
        response_ids = self._text_tokenizer(
            response, return_tensors="pt").input_ids
        # ``input_ids`` is a tensor for this tokenizer but a plain list for some
        # slow tokenizers; len() works for both.
        return (len(prompt_ids[0]), len(response_ids[0]))

    def generate_text(self, prompt, previous_image, max_tokens, temperature):
        if previous_image is not None:
            return self._answer_image_question(
                previous_image, prompt, max_tokens, temperature
            )
        conversation = self._text_conversation.copy()
        conversation.append_message(conversation.roles[0], prompt)
        conversation.append_message(conversation.roles[1], None)
        input_ids = self._text_tokenizer(
            conversation.get_prompt(), return_tensors="pt"
        ).input_ids.to(self._text_model.device)
        generation_kwargs = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_tokens,
            "use_cache": True,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with self._lock, torch.inference_mode():
            output_ids = self._text_model.generate(
                input_ids,
                **generation_kwargs,
            )
        return self._decode_new_tokens(output_ids, input_ids)

    def _decode_new_tokens(self, output_ids, prompt_ids):
        """Decode a LLaVA ``generate`` result whose first column is new text."""
        minimum_prefix = 8
        if (
            prompt_ids.shape[1] >= minimum_prefix
            and output_ids.shape[1] > prompt_ids.shape[1]
            and torch.equal(output_ids[0, : prompt_ids.shape[1]], prompt_ids[0])
        ):
            output_ids = output_ids[:, prompt_ids.shape[1]:]
        return self._text_tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

    def _answer_image_question(self, image, question, max_tokens, temperature):
        """Official TokenFlow i2t recipe from i2t/llava/serve/model_worker.py."""
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        conversation = self._text_conversation.copy()
        conversation.append_message(
            conversation.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        )
        conversation.append_message(conversation.roles[1], None)
        input_ids = tokenizer_image_token(
            conversation.get_prompt(),
            self._text_tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self._text_model.device)
        pixels = process_images(
            [image], self._review_image_processor, self._text_model.config
        )
        if isinstance(pixels, list):
            pixels = [
                tensor.to(device=self._text_model.device, dtype=self._text_dtype)
                for tensor in pixels
            ]
        else:
            pixels = pixels.to(
                device=self._text_model.device, dtype=self._text_dtype
            )
        generation_kwargs = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_tokens,
            "use_cache": True,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with self._lock, torch.inference_mode():
            output_ids = self._text_model.generate(
                inputs=input_ids, images=pixels, **generation_kwargs
            )
        return self._decode_new_tokens(output_ids, input_ids)

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
        del previous_image, previous_images, image_steps
        conditional, _ = self._crop_and_encode(
            self._image_tokenizer,
            content,
            image=None,
            max_text_token_num=128,
        )
        unconditional, _ = self._crop_and_encode(
            self._image_tokenizer,
            self._negative_prompt,
            image=None,
            max_text_token_num=128,
        )
        image_device = self._image_model.get_model().embed_tokens.weight.device
        conditional = conditional.to(image_device)
        unconditional = unconditional.to(image_device)
        with self._lock:
            samples = self._image_model.autoregressive_infer_cfg(
                B=1,
                prefix_text_codes=[conditional, unconditional],
                cfg=7.5,
                topk_list=[600],
                topp_list=[0.6],
                g_seed=None if seed is None else int(seed),
            )
        sample = samples[0]
        if isinstance(sample, torch.Tensor):
            sample = sample.float().cpu().numpy()
        array = np.asarray(sample)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = array.transpose(1, 2, 0)
        if array.max() <= 1.0:
            array = array * 255
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        return image.resize((image_shape[1], image_shape[0]), pil_lanczos())


def load_tokenflow_backend(args):
    if not args.vq_path:
        raise SystemExit("--vq-path is required for --backend tokenflow")
    if not args.llm_path:
        raise SystemExit("--llm-path is required for --backend tokenflow")
    return TokenFlowBackend(
        args.repo_path,
        args.model_path,
        args.vq_path,
        args.llm_path,
        args.device,
        getattr(args, "understanding_vision_path", None),
    )
