"""Show-o2 7B adapter for the unified UMM protocol."""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock

import torch
from PIL import Image

from baselines.umm_framework import (
    BackendCapabilities,
    StatelessMinimalBackend,
    _normalise_previous_images,
    seeded_generator,
)
from baselines.umm_model_identity import model_artifact_identity


def _decode_wan_single_frame(vae, samples):
    """Apply the official Show-o2 single-frame WanVAE shape conversion."""
    decoded = vae.batch_decode(samples.unsqueeze(2))
    if decoded.ndim != 5 or decoded.shape[2] != 1:
        raise RuntimeError(
            "Show-o2 WanVAE returned an unexpected decoded video shape: "
            f"{tuple(decoded.shape)}"
        )
    return decoded.squeeze(2)


class Showo2Backend(StatelessMinimalBackend):
    name = "show-o2-7b"
    capabilities = BackendCapabilities(
        name=name,
        condition_images_consumed=True,
        previous_image_consumed=True,
        multiple_condition_images_consumed=True,
        requested_shape_consumed=False,
          requested_image_steps_consumed=True,
          requested_seed_consumed=True,
          reviewer_supported=True,
          planner_transport="official Show-o2 MMU text head",
          planner_context_limit=32768,
        image_config={
            "sampler": "official Show-o2 flow-matching T2I head",
            "native_resolution": [432, 432],
            "llm_parameters": "7B",
        },
    )

    def __init__(self, repo_path: str, model_path: str, vae_path: str, llm_path: str):
        repo = Path(repo_path).resolve()
        if str(repo / "show-o2") not in sys.path:
            sys.path.insert(0, str(repo / "show-o2"))
        from models import Showo2Qwen2_5, WanVAE
        from models.misc import get_text_tokenizer
        from omegaconf import OmegaConf
        from utils import get_hyper_params, path_to_llm_name

        config = OmegaConf.load(
            repo / "show-o2/configs/showo2_7b_demo_432x432.yaml")
        config.model.showo.pretrained_model_path = str(Path(model_path).resolve())
        config.model.showo.llm_model_path = str(Path(llm_path).resolve())
        config.model.vae_model.pretrained_model_path = str(Path(vae_path).resolve())
        device = torch.device("cuda")
        self._device = device
        self._weight_type = torch.bfloat16
        self._tokenizer, self._token_ids = get_text_tokenizer(
            config.model.showo.llm_model_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name=path_to_llm_name["Qwen/Qwen2.5-7B-Instruct"],
        )
        config.model.showo.llm_vocab_size = len(self._tokenizer)
        self._model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path,
            use_safetensors=False,
        ).to(device=device, dtype=self._weight_type).eval()
        if config.model.showo.add_time_embeds:
            config.dataset.preprocessing.num_t2i_image_tokens += 1
            config.dataset.preprocessing.num_mmu_image_tokens += 1
        self._hyper = get_hyper_params(
            config, self._tokenizer, self._token_ids)
        (
            self._num_t2i_tokens,
            self._num_mmu_tokens,
            _num_video_tokens,
            self._max_seq_len,
            _max_text_len,
            self._image_latent_dim,
            self._patch_size,
            self._latent_width,
            self._latent_height,
            self._pad_id,
            self._bos_id,
            self._eos_id,
            self._boi_id,
            self._eoi_id,
            _bov_id,
            _eov_id,
            _img_pad_id,
            _vid_pad_id,
            self._guidance_scale,
        ) = self._hyper
        self._configured_vae_path = str(Path(vae_path).resolve())
        self._resolution = int(config.dataset.preprocessing.resolution)
        self._vae = WanVAE(
            vae_pth=self._configured_vae_path,
            dtype=self._weight_type,
            device=device,
        )
        self._generation_lock = Lock()
        self.model_identity = model_artifact_identity(
            model_path,
            extra_artifacts={
                "llm": llm_path,
                "vae": vae_path,
            },
        )

    def planner_token_counts(self, system_text, user_text, response):
        """Planner prompt/response token counts for this backend."""
        del system_text
        prompt_ids = self._tokenizer(user_text, add_special_tokens=False).input_ids
        response_ids = self._tokenizer(
            response, add_special_tokens=False).input_ids
        return (len(prompt_ids), len(response_ids))

    def generate_text(self, prompt, previous_image, max_tokens, temperature):
        from models import omni_attn_mask_naive

        with self._generation_lock:
            if previous_image is not None:
                return self._answer_image_question(
                    previous_image, prompt, max_tokens, omni_attn_mask_naive)
            return self._generate_text_locked(
                prompt, max_tokens, omni_attn_mask_naive)

    def _answer_image_question(
        self, image, question_text, max_tokens, omni_attn_mask_naive,
    ):
        """Official Show-o2 MMU recipe from show-o2/inference_mmu.py."""
        from datasets.utils import image_transform

        with torch.inference_mode():
            tensor = image_transform(
                image, resolution=self._resolution
            ).to(self._device)
            latents = self._vae.sample(
                tensor.unsqueeze(0).unsqueeze(2)
            ).squeeze(2).to(self._weight_type)
            embeds_und = self._model.image_embedder_und(latents)
            embeds_gen = self._model.image_embedder_gen(latents)
            embeds_und = embeds_und + self._model.position_embedding(
                self._model.image_position_ids)
            embeds_und = self._model.und_trans(
                embeds_und)["last_hidden_state"]
            image_embeds = self._model.fusion_proj(
                torch.cat([embeds_und, embeds_gen], dim=-1))

            system_ids = self._tokenizer(
                "system\nYou are a helpful assistant.<|im_end|>",
                add_special_tokens=False,
            )["input_ids"]
            role_user = self._tokenizer(
                "\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
            role_assistant = self._tokenizer(
                "\n<|im_start|>assistant\n",
                add_special_tokens=False,
            )["input_ids"]
            question = self._tokenizer(
                question_text, add_special_tokens=False).input_ids
            prefix = torch.tensor(
                [self._token_ids["bos_id"]] + system_ids + role_user,
                device=self._device,
            )[None]
            suffix = torch.tensor(
                [self._token_ids["boi_id"], self._token_ids["eoi_id"]]
                + question + role_assistant,
                device=self._device,
            )[None]
            embeds_a = self._model.showo.model.embed_tokens(prefix)
            embeds_b = self._model.showo.model.embed_tokens(suffix)
            time_embed = self._model.time_embed(
                torch.tensor([[1.0]], device=self._device), embeds_a.dtype)
            if hasattr(self._model, "time_embed_proj"):
                time_embed = self._model.time_embed_proj(time_embed)
            input_embeds = torch.cat(
                [
                    embeds_a,
                    embeds_b[:, :1],
                    time_embed,
                    image_embeds,
                    embeds_b[:, 1:],
                ],
                dim=1,
            ).to(self._weight_type)
            positions = torch.tensor(
                [prefix.shape[1] + 2, self._num_mmu_tokens],
                device=self._device,
            )[None, None]
            attention = omni_attn_mask_naive(
                B=input_embeds.size(0),
                LEN=input_embeds.shape[1],
                modalities=positions,
                device=self._device,
                inverted=True,
            ).to(input_embeds.dtype)
            output_tokens = self._model.mmu_generate(
                input_embeds=input_embeds,
                attention_mask=attention,
                top_k=1,
                max_new_tokens=max_tokens,
                eos_token=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            output_tokens, skip_special_tokens=True
        ).strip()

    def _generate_text_locked(self, prompt, max_tokens, omni_attn_mask_naive):
        system_ids = self._tokenizer(
            "system\nYou are a helpful assistant.<|im_end|>",
            add_special_tokens=False,
        )["input_ids"]
        role_user = self._tokenizer(
            "\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        role_assistant = self._tokenizer(
            "\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        question = self._tokenizer(prompt, add_special_tokens=False).input_ids
        prefix = torch.tensor(
            [self._token_ids["bos_id"]] + system_ids + role_user,
            device=self._device,
        )[None]
        suffix = torch.tensor(
            [self._token_ids["boi_id"], self._token_ids["eoi_id"]]
            + question + role_assistant,
            device=self._device,
        )[None]
        embeds_a = self._model.showo.model.embed_tokens(prefix)
        embeds_b = self._model.showo.model.embed_tokens(suffix)
        time_embed = self._model.time_embed(
            torch.tensor([[1.0]], device=self._device), embeds_a.dtype)
        if hasattr(self._model, "time_embed_proj"):
            time_embed = self._model.time_embed_proj(time_embed)
        input_embeds = torch.cat(
            [embeds_a, embeds_b[:, :1], time_embed, embeds_b[:, 1:]], dim=1)
        positions = torch.tensor(
            [prefix.shape[1] + 2, self._num_mmu_tokens],
            device=self._device,
        )[None, None]
        attention = omni_attn_mask_naive(
            1, input_embeds.shape[1], positions, self._device
        ).to(input_embeds.dtype)
        output_tokens = self._model.mmu_generate(
            input_embeds=input_embeds,
            attention_mask=attention,
            top_k=1,
            max_new_tokens=max_tokens,
            eos_token=self._tokenizer.eos_token_id,
        )
        return self._tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def generate_image(
        self, content, previous_image, seed, image_shape, image_steps,
        previous_images=None,
    ):
        from models.misc import prepare_gen_input
        from transport import Sampler, create_transport
        from utils import denorm

        sources = _normalise_previous_images(previous_image, previous_images)
        with self._generation_lock:
            if sources:
                return self._generate_image_with_condition(
                    content, sources, seed, image_shape, image_steps,
                    prepare_gen_input, Sampler, create_transport, denorm)
            return self._generate_image_locked(
                content, seed, image_shape, image_steps,
                prepare_gen_input, Sampler, create_transport, denorm)

    def _mixed_modal_tokens(self, story_text, source_count, num_tokens):
        """Lay out one VIST-style story prefix plus the image being denoised."""
        tokens = [self._bos_id]
        positions = []
        if story_text:
            tokens.extend(
                self._tokenizer(story_text, add_special_tokens=False).input_ids
            )
        for _ in range(source_count):
            positions.append((len(tokens) + 1, num_tokens))
            tokens.extend(
                [self._boi_id] + [self._hyper[16]] * num_tokens
                + [self._eoi_id]
            )
        positions.append((len(tokens) + 1, num_tokens))
        tokens.extend(
            [self._boi_id] + [self._hyper[16]] * num_tokens + [self._eoi_id]
        )
        tokens.append(self._eos_id)
        return tokens, positions

    def _generate_image_with_condition(
        self, content, sources, seed, image_shape, image_steps,
        prepare_gen_input, Sampler, create_transport, denorm,
    ):
        """Official Show-o2 mixed-modality recipe from inference_mixed_modality."""
        del prepare_gen_input
        from datasets.utils import image_transform
        from models import omni_attn_mask_naive

        num_tokens = self._num_t2i_tokens
        source_latents = [
            self._vae.sample(
                image_transform(source, resolution=self._resolution)
                .to(self._device)
                .unsqueeze(0)
                .unsqueeze(2)
            )
            .squeeze(2)
            .to(self._weight_type)
            for source in sources
        ]

        cond_tokens, cond_positions = self._mixed_modal_tokens(
            content, len(sources), num_tokens)
        null_tokens, _null_positions = self._mixed_modal_tokens(
            "", 0, num_tokens)
        length = max(len(cond_tokens), len(null_tokens))
        if length % 128:
            length = (length // 128 + 1) * 128
        cond_tokens = cond_tokens + [self._pad_id] * (length - len(cond_tokens))
        null_tokens = null_tokens + [self._pad_id] * (length - len(null_tokens))
        text_tokens = torch.tensor(
            [cond_tokens, null_tokens], device=self._device
        )
        # Every null-branch image shares the single leading block position, the
        # same convention inference_mixed_modality.py uses for its null history.
        null_positions = torch.zeros(
            (1, len(cond_positions), 2), device=self._device, dtype=torch.long
        )
        null_positions[:] = torch.tensor(
            [2, num_tokens], device=self._device
        )
        positions = torch.cat(
            [
                torch.tensor([cond_positions], device=self._device,
                             dtype=torch.long),
                null_positions,
            ],
            dim=0,
        )

        latent_shape = (
            1, self._image_latent_dim,
            self._latent_height * self._patch_size,
            self._latent_width * self._patch_size,
        )
        generator = seeded_generator(self._device, seed)
        noise = torch.randn(
            latent_shape, generator=generator, device=self._device,
            dtype=self._weight_type,
        )
        z = torch.cat([*source_latents, noise], dim=0)
        z_null = noise.clone().repeat(len(sources) + 1, 1, 1, 1)
        z = torch.cat([z, z_null], dim=0)

        transport = create_transport(
            path_type="Linear", prediction="velocity", loss_weight=None,
            train_eps=None, sample_eps=None, snr_type="lognorm",
            do_shift=True, seq_len=num_tokens)
        sampler = Sampler(transport)
        sample_fn = sampler.sample_ode(
            sampling_method="euler", num_steps=image_steps,
            atol=1e-6, rtol=1e-3, reverse=False, time_shifting_factor=3.0)
        attention = omni_attn_mask_naive(
            # The official mixed-modality loop builds the block mask over the
            # padded token tensor itself, not over the configured max sequence.
            text_tokens.shape[0], text_tokens.shape[1], positions, self._device
        ).to(self._weight_type)
        samples = sample_fn(
            z, self._model.t2i_generate, text_tokens=text_tokens,
            attention_mask=attention, modality_positions=positions,
            output_hidden_states=True, max_seq_len=self._max_seq_len,
            guidance_scale=self._guidance_scale,
            only_denoise_last_image=True,
        )[-1]
        samples = torch.chunk(samples, 2)[0]
        decoded = _decode_wan_single_frame(self._vae, samples[-1:])
        image = Image.fromarray(denorm(decoded)[0])
        return image.resize((image_shape[1], image_shape[0]))

    def _generate_image_locked(
        self, content, seed, image_shape, image_steps,
        prepare_gen_input, Sampler, create_transport, denorm,
    ):
        prompts = [content]
        tokens = prepare_gen_input(
            prompts, self._tokenizer, self._num_t2i_tokens,
            self._bos_id, self._eos_id, self._boi_id, self._eoi_id,
            self._pad_id, self._hyper[16], self._hyper[4], self._device)
        text_tokens, null_tokens, positions, null_positions = tokens
        latent_shape = (
            1, self._image_latent_dim,
            self._latent_height * self._patch_size,
            self._latent_width * self._patch_size,
        )
        generator = seeded_generator(self._device, seed)
        z = torch.randn(latent_shape, generator=generator, device=self._device,
                        dtype=self._weight_type)
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([text_tokens, null_tokens], dim=0)
        positions = torch.cat([positions, null_positions], dim=0)
        transport = create_transport(
            path_type="Linear", prediction="velocity", loss_weight=None,
            train_eps=None, sample_eps=None, snr_type="lognorm",
            do_shift=True, seq_len=self._num_t2i_tokens)
        sampler = Sampler(transport)
        sample_fn = sampler.sample_ode(
            sampling_method="euler", num_steps=image_steps,
            atol=1e-6, rtol=1e-3, reverse=False, time_shifting_factor=3.0)
        from models import omni_attn_mask_naive
        attention = omni_attn_mask_naive(
            text_tokens.shape[0], self._max_seq_len, positions, self._device
        ).to(self._weight_type)
        samples = sample_fn(
            z, self._model.t2i_generate, text_tokens=text_tokens,
            attention_mask=attention, modality_positions=positions,
            output_hidden_states=True, max_seq_len=self._max_seq_len,
            guidance_scale=self._guidance_scale)[-1]
        samples = torch.chunk(samples, 2)[0]
        decoded = _decode_wan_single_frame(self._vae, samples)
        image = Image.fromarray(denorm(decoded)[0])
        return image.resize((image_shape[1], image_shape[0]))


def load_showo2_backend(args):
    backend = Showo2Backend(
        args.repo_path, args.model_path, args.vq_path, args.llm_path)
    return backend
