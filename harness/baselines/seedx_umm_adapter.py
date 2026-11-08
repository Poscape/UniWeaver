"""SEED-X 17B adapter for the unified UMM protocol."""

from __future__ import annotations

import re
import sys
import types
from contextlib import nullcontext
from importlib.machinery import ModuleSpec
from pathlib import Path
from threading import Lock

import torch

from baselines.umm_framework import (
    BackendCapabilities,
    StatelessMinimalBackend,
    _normalise_previous_images,
)
from baselines.umm_model_identity import model_artifact_identity


class SeedXBackend(StatelessMinimalBackend):
    name = "seed-x-17b"
    capabilities = BackendCapabilities(
        name=name,
        condition_images_consumed=True,
        previous_image_consumed=True,
        multiple_condition_images_consumed=False,
        requested_shape_consumed=True,
          requested_image_steps_consumed=True,
          requested_seed_consumed=True,
          reviewer_supported=True,
          planner_transport="official SEED-X 17B ContinuousLVLM text head",
          planner_context_limit=2048,
        image_config={
            "sampler": "official SEED-X SDXL adapter",
            "guidance_scale": 7.5,
            "image_condition_tokens": 64,
        },
    )

    def __init__(
        self,
        repo_path: str,
        model_path: str,
        visual_encoder_path: str,
        sdxl_path: str,
        device: str,
    ):
        repo = Path(repo_path).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import hydra
        # SEED-X imports DeepSpeed and transformers.deepspeed at module load.
        # Provide the optional shim when the package is not otherwise present,
        # while keeping the installed Torch stack unchanged.
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            deepspeed_stub = types.ModuleType("deepspeed")
            deepspeed_stub.__spec__ = ModuleSpec("deepspeed", loader=None)
            deepspeed_stub.zero = types.SimpleNamespace(
                GatheredParameters=lambda *args, **kwargs: nullcontext()
            )
            sys.modules["deepspeed"] = deepspeed_stub
        try:
            import transformers.deepspeed  # noqa: F401
        except ImportError:
            transformers_deepspeed = types.ModuleType("transformers.deepspeed")
            transformers_deepspeed.is_deepspeed_zero3_enabled = lambda: False
            sys.modules["transformers.deepspeed"] = transformers_deepspeed
        try:
            import xformers.ops  # noqa: F401
        except ImportError:
            xformers_stub = types.ModuleType("xformers")
            xops_stub = types.ModuleType("xformers.ops")
            xformers_stub.__spec__ = ModuleSpec("xformers", loader=None)
            xops_stub.__spec__ = ModuleSpec("xformers.ops", loader=None)

            class LowerTriangularMask:
                pass

            def memory_efficient_attention(query, key, value, attn_bias=None):
                output = torch.nn.functional.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    is_causal=isinstance(attn_bias, LowerTriangularMask),
                )
                return output.transpose(1, 2)

            xops_stub.LowerTriangularMask = LowerTriangularMask
            xops_stub.memory_efficient_attention = memory_efficient_attention
            xformers_stub.ops = xops_stub
            sys.modules["xformers"] = xformers_stub
            sys.modules["xformers.ops"] = xops_stub
        from diffusers import (
            AutoencoderKL,
            EulerDiscreteScheduler,
            UNet2DConditionModel,
        )
        from omegaconf import OmegaConf

        self._device = torch.device(device)
        self._dtype = torch.float16
        self._lock = Lock()
        root = Path(model_path).resolve()

        tokenizer_cfg = OmegaConf.load(
            repo / "configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml"
        )
        tokenizer_cfg.pretrained_model_name_or_path = str(
            root / "cvlm_llama2_tokenizer_100img_and_224loc_addpatch"
        )
        self._tokenizer = hydra.utils.instantiate(tokenizer_cfg)

        visual_cfg = OmegaConf.load(
            repo / "configs/visual_encoder/qwen_vitg_448.yaml"
        )
        visual_cfg.pretrained_model_path = str(Path(visual_encoder_path).resolve())
        visual_encoder = hydra.utils.instantiate(visual_cfg).eval().to(
            self._device, dtype=self._dtype
        )

        llm_cfg = OmegaConf.load(repo / "configs/clm_models/llm_seed_x.yaml")
        llm_cfg.pretrained_model_name_or_path = str(root / "seed_x/llm")
        llm = hydra.utils.instantiate(llm_cfg, torch_dtype=self._dtype)
        agent_cfg = OmegaConf.load(repo / "configs/clm_models/agent_seed_x.yaml")
        agent_cfg.pretrained_model_path = str(
            root / "seed_x/agent/pytorch_model.bin"
        )
        self._agent = hydra.utils.instantiate(agent_cfg, llm=llm).eval().to(
            self._device, dtype=self._dtype
        )

        scheduler = EulerDiscreteScheduler.from_pretrained(
            sdxl_path, subfolder="scheduler"
        )
        vae = AutoencoderKL.from_pretrained(
            sdxl_path, subfolder="vae"
        ).to(self._device, dtype=self._dtype)
        unet = UNet2DConditionModel.from_pretrained(
            sdxl_path, subfolder="unet"
        ).to(self._device, dtype=self._dtype)
        adapter_cfg = OmegaConf.load(
            repo
            / "configs/sdxl_adapter/"
            "sdxl_qwen_vit_resampler_l4_q64_pretrain_no_normalize.yaml"
        )
        adapter_cfg.pretrained_model_path = str(
            root / "seed_detokenizer/first_stage/pytorch_model.bin"
        )
        self._adapter = hydra.utils.instantiate(
            adapter_cfg, unet=unet
        ).to(self._device, dtype=self._dtype).eval()
        discrete_cfg = OmegaConf.load(
            repo / "configs/discrete_model/discrete_identity.yaml"
        )
        discrete_model = hydra.utils.instantiate(discrete_cfg).to(self._device).eval()
        image_transform = hydra.utils.instantiate(
            OmegaConf.load(repo / "configs/processer/qwen_448_transform.yaml")
        )
        self._adapter.init_pipe(
            vae=vae,
            scheduler=scheduler,
            visual_encoder=visual_encoder,
            image_transform=image_transform,
            discrete_model=discrete_model,
            dtype=self._dtype,
            device=self._device,
        )
        # Retained for the Plan4 review (image -> text) path.
        self._repo = repo
        self._model_root = root
        self._sdxl_path = sdxl_path
        self._visual_encoder = visual_encoder
        self._edit_stack = None
        self._image_transform = image_transform
        self._grid_pinpoints = [
            [int(rows) * 448, int(cols) * 448]
            for rows, cols in (
                scale.split("x") for scale in (
                    "1x1", "1x2", "1x3", "2x1", "3x1", "1x4", "4x1", "2x2"
                )
            )
        ]
        self._special_token_ids = {
            # The last id is the real marker; [0] is the SentencePiece
            # prefix shared by every marker, which makes the span empty.
            name: self._tokenizer.encode(token, add_special_tokens=False)[-1]
            for name, token in (
              ("boi", "<img>"), ("bop", "<patch>"),
              ("eoi", "</img>"), ("eop", "</patch>"),
          )
      }
        self.model_identity = model_artifact_identity(
            model_path,
            extra_artifacts={
                "visual_encoder": visual_encoder_path,
                "sdxl": sdxl_path,
            },
        )

    def planner_token_counts(self, system_text, user_text, response):
        """Planner prompt/response token counts for this backend."""
        del system_text
        prompt = f"Question: {user_text}\nAnswer:"
        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self._tokenizer.encode(response, add_special_tokens=False)
        return (len(prompt_ids), len(response_ids))

    def generate_text(self, prompt, previous_image, max_tokens, temperature):
        if previous_image is not None:
            return self._answer_image_question(previous_image, prompt, max_tokens)
        text = f"Question: {prompt}\nAnswer:"
        input_ids = self._tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor(
            [self._tokenizer.bos_token_id, *input_ids],
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0)
        generation_kwargs = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_tokens,
            "use_cache": True,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with self._lock, torch.inference_mode():
            output_ids = self._agent.llm.generate(
                input_ids=input_ids,
                **generation_kwargs,
            )
        generated = output_ids[0, input_ids.shape[1]:]
        output = self._tokenizer.decode(generated, skip_special_tokens=True)
        return re.sub(r"<[^>]*>", "", output).strip()

    def _answer_image_question(self, image, question, max_tokens):
        """Official SEED-X image->text recipe from eval_img2text_seed_x_i.py."""
        inference_dir = self._repo / "src" / "inference"
        if str(inference_dir) not in sys.path:
            sys.path.insert(0, str(inference_dir))
        from any_res import process_anyres_image

        image_tensor, patch_positions = process_anyres_image(
            image, self._image_transform, self._grid_pinpoints, 448
        )
        embeds_cmp_mask = torch.tensor(
            [True] * image_tensor.shape[0], device=self._device, dtype=torch.bool
        )
        patch_position = torch.cat([patch_positions], dim=0).to(self._device)
        image_tensor = image_tensor.to(self._device, dtype=self._dtype)

        per_image = "".join(
            f"<img_{index:05d}>" for index in range(64)
        )
        image_tokens = "".join(
            f"<patch>{per_image}</patch>" for _ in range(image_tensor.shape[0] - 1)
        )
        image_tokens += f"<img>{per_image}</img>"

        text = f"[INST] {image_tokens}{question} [/INST]\n"
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        token_ids = torch.tensor(
            [self._tokenizer.bos_token_id, *token_ids],
            dtype=torch.long,
            device=self._device,
        )
        compare_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        boi = self._special_token_ids
        start = torch.where(
            (token_ids == boi["boi"]) | (token_ids == boi["bop"])
        )[0].tolist()
        end = torch.where(
            (token_ids == boi["eoi"]) | (token_ids == boi["eop"])
        )[0].tolist()
        for begin, finish in zip(start, end):
            compare_mask[begin + 1:finish] = True

        with self._lock, torch.inference_mode():
            image_embeds = self._visual_encoder(image_tensor)
            output = self._agent.generate(
                tokenizer=self._tokenizer,
                input_ids=token_ids.unsqueeze(0),
                image_embeds=image_embeds,
                embeds_cmp_mask=embeds_cmp_mask,
                patch_positions=patch_position,
                ids_cmp_mask=compare_mask.unsqueeze(0),
                max_new_tokens=max_tokens,
                num_img_gen_tokens=64,
            )
        return re.sub(r"<[^>]*>", "", str(output.get("text", ""))).strip()

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

    def _ensure_edit_stack(self):
        """Load the released editing LLM/agent and second-stage detokenizer once."""
        if self._edit_stack is not None:
            return self._edit_stack
        import hydra
        from diffusers import (
            AutoencoderKL,
            EulerDiscreteScheduler,
            UNet2DConditionModel,
        )
        from omegaconf import OmegaConf

        root = self._model_root
        edit_llm = root / "seed_x_edit/llm"
        edit_agent = root / "seed_x_edit/agent/pytorch_model.bin"
        second_stage = root / "seed_detokenizer/second_stage/pytorch_model.bin"
        missing = [
            str(path) for path in (edit_llm, edit_agent, second_stage)
            if not path.exists()
        ]
        if missing:
            # Runtime, not SystemExit: this is discovered while a task is
            # running, and the runner's task boundary only classifies
            # exceptions (SystemExit would escape and kill the process).
            raise RuntimeError(
                "SEED-X image editing needs the released edit subset "
                f"(seed_x_edit/*, seed_detokenizer/second_stage/*): missing {missing}"
            )

        llm_cfg = OmegaConf.load(
            self._repo / "configs/clm_models/llm_seed_x_edit.yaml"
        )
        llm_cfg.pretrained_model_name_or_path = str(edit_llm)
        edit_llm_model = hydra.utils.instantiate(llm_cfg, torch_dtype=self._dtype)
        agent_cfg = OmegaConf.load(
            self._repo / "configs/clm_models/agent_seed_x_edit.yaml"
        )
        agent_cfg.pretrained_model_path = str(edit_agent)
        edit_agent_model = hydra.utils.instantiate(
            agent_cfg, llm=edit_llm_model
        ).eval().to(self._device, dtype=self._dtype)

        scheduler = EulerDiscreteScheduler.from_pretrained(
            self._sdxl_path, subfolder="scheduler"
        )
        vae = AutoencoderKL.from_pretrained(
            self._sdxl_path, subfolder="vae"
        ).to(self._device, dtype=self._dtype)
        unet = UNet2DConditionModel.from_pretrained(
            self._sdxl_path, subfolder="unet"
        ).to(self._device, dtype=self._dtype)
        adapter_cfg = OmegaConf.load(
            self._repo
            / "configs/sdxl_adapter/"
            "sdxl_qwen_vit_resampler_l4_q64_full_with_latent_image_pretrain_no_normalize.yaml"
        )
        adapter_cfg.pretrained_model_path = str(second_stage)
        edit_adapter = hydra.utils.instantiate(
            adapter_cfg, unet=unet
        ).to(self._device, dtype=self._dtype).eval()
        # The editing adapter rewrites the UNet input convolution for the
        # concatenated latent image, so it must own an UNet of its own.
        edit_adapter.init_pipe(
            vae=vae,
            scheduler=scheduler,
            visual_encoder=self._visual_encoder,
            image_transform=self._image_transform,
            dtype=self._dtype,
            device=self._device,
        )
        self._edit_stack = (edit_agent_model, edit_adapter)
        return self._edit_stack

    def _image_instruction_tokens(self, image, instruction):
        """Encode one source frame with the official SEED-X image-token layout."""
        inference_dir = self._repo / "src" / "inference"
        if str(inference_dir) not in sys.path:
            sys.path.insert(0, str(inference_dir))
        from any_res import process_anyres_image

        image_tensor, patch_positions = process_anyres_image(
            image, self._image_transform, self._grid_pinpoints, 448
        )
        embeds_cmp_mask = torch.tensor(
            [True] * image_tensor.shape[0], device=self._device, dtype=torch.bool
        )
        patch_position = torch.cat([patch_positions], dim=0).to(self._device)
        image_tensor = image_tensor.to(self._device, dtype=self._dtype)
        per_image = "".join(f"<img_{index:05d}>" for index in range(64))
        image_tokens = "".join(
            f"<patch>{per_image}</patch>"
            for _ in range(image_tensor.shape[0] - 1)
        )
        image_tokens += f"<img>{per_image}</img>"
        text = f"[INST] {image_tokens}{instruction} [/INST]\n"
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        token_ids = torch.tensor(
            [self._tokenizer.bos_token_id, *token_ids],
            dtype=torch.long,
            device=self._device,
        )
        compare_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        markers = self._special_token_ids
        starts = torch.where(
            (token_ids == markers["boi"]) | (token_ids == markers["bop"])
        )[0].tolist()
        ends = torch.where(
            (token_ids == markers["eoi"]) | (token_ids == markers["eop"])
        )[0].tolist()
        for begin, finish in zip(starts, ends):
            compare_mask[begin + 1:finish] = True
        return (
            image_tensor,
            embeds_cmp_mask,
            patch_position,
            token_ids.unsqueeze(0),
            compare_mask.unsqueeze(0),
        )

    def _edit_image(self, source, instruction, seed, image_shape, image_steps):
        """Official SEED-X image-editing recipe from eval_img2edit_seed_x_edit.py."""
        agent, adapter = self._ensure_edit_stack()
        (
            image_tensor,
            embeds_cmp_mask,
            patch_position,
            token_ids,
            compare_mask,
        ) = self._image_instruction_tokens(source, instruction)
        image_token_ids = [
            self._tokenizer.convert_tokens_to_ids(f"<img_{index:05d}>")
            for index in range(63)
        ]
        forced_ids = (
            token_ids[0].tolist()
            + [self._special_token_ids["boi"]]
            + image_token_ids
        )
        input_ids = torch.tensor(
            forced_ids, dtype=torch.long, device=self._device
        ).unsqueeze(0)
        with self._lock:
            input_embeds = agent.llm.get_input_embeddings()(input_ids)
            image_embeds = self._visual_encoder(image_tensor)
            image_embeds_lm = agent.input_resampler(image_embeds)
            if getattr(agent, "add_patch_pos", False):
                positions = patch_position.to(image_embeds_lm)
                relative = torch.mm(
                    torch.cat([positions, 1 - positions], dim=-1) / 2,
                    agent.patch_pos_embed,
                ).unsqueeze(1)
                image_embeds_lm = image_embeds_lm + relative
            prefix = input_embeds[:, : compare_mask.shape[1]]
            prefix[0][compare_mask[0]] = image_embeds_lm[embeds_cmp_mask].view(
                -1, input_embeds.shape[-1]
            )
            output = agent.llm(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                return_dict=True,
            )
            image_states = output.hidden_states[-1][:, -64:, :]
            image_embedding = agent.output_resampler(image_states)
        latent_image = source.convert("RGB").resize((1024, 1024))
        images = adapter.generate(
            image_embeds=image_embedding.to(self._device),
            latent_image=latent_image,
            seed=None if seed is None else int(seed),
            height=image_shape[0],
            width=image_shape[1],
            guidance_scale=7.5,
            num_inference_steps=image_steps,
        )
        if not images:
            raise RuntimeError("SEED-X editing adapter returned no image")
        return images[0]

    def _text_to_image(self, content, seed, image_shape, image_steps):
        prompt_ids = self._tokenizer.encode(
            f"{content}<img>", add_special_tokens=False
        )
        # The official LogitsProcessor deterministically forces these tokens
        # after <img>. A single causal forward over the forced prefix produces
        # the 64 states (BOI + image tokens 0-62) used by the output resampler.
        image_token_ids = [
            self._tokenizer.convert_tokens_to_ids(f"<img_{index:05d}>")
            for index in range(63)
        ]
        input_ids = torch.tensor(
            [self._tokenizer.bos_token_id, *prompt_ids, *image_token_ids],
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0)
        with self._lock:
            output = self._agent.llm(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            image_states = output.hidden_states[-1][:, -64:, :]
            image_embedding = self._agent.output_resampler(image_states)
            images = self._adapter.generate(
                image_embeds=image_embedding.to(self._device),
                seed=int(seed),
                height=image_shape[0],
                width=image_shape[1],
                guidance_scale=7.5,
                num_inference_steps=image_steps,
            )
        if not images:
            raise RuntimeError("SEED-X SDXL adapter returned no image")
        return images[0]


def load_seedx_backend(args):
    if not args.vq_path:
        raise SystemExit("--vq-path is required for --backend seed-x")
    if not args.llm_path:
        raise SystemExit("--llm-path is required for --backend seed-x")
    return SeedXBackend(
        args.repo_path,
        args.model_path,
        args.vq_path,
        args.llm_path,
        args.device,
    )
