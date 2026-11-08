"""Explicit UniWeaver behavior layered over the unmodified BAGEL inferencer.

This module does not construct Plan1-4 and is not a selectable Planner path.
"""

from copy import deepcopy

import torch
from PIL import Image

from bagel_multimodal_text_generation import generate_text
from data.data_utils import pil_img2rgb
from inferencer import (
    GEN_THINK_SYSTEM_PROMPT,
    VLM_THINK_SYSTEM_PROMPT,
    InterleaveInferencer,
)
from runtime_diagnostics import runtime_diagnostic as _diag


def _diagnose_tensor(label, tensor):
    """Record numerical corruption without changing upstream model code."""
    if tensor.isnan().any() or tensor.isinf().any():
        _diag(
            f'{label} NaN={tensor.isnan().any().item()} '
            f'Inf={tensor.isinf().any().item()} shape={tuple(tensor.shape)}')


class UniWeaverInterleaveInferencer(InterleaveInferencer):
    """Add only reproducibility, reference sizing and diagnostics."""

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",

        num_timesteps=50,
        timestep_shift=3.0,
        enable_taylorseer=False,
        image_seed=None,
    ):
        import sys
        import time
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        if image_seed is not None:
            # Preserve the upstream method and replace only its random result.
            upstream_noise = generation_input['packed_init_noises']
            noise_generator = torch.Generator(device=upstream_noise.device)
            noise_generator.manual_seed(int(image_seed))
            generation_input['packed_init_noises'] = torch.randn(
                upstream_noise.shape, dtype=upstream_noise.dtype,
                device=upstream_noise.device, generator=noise_generator)
        _diagnose_tensor(
            'PREPARED_IMAGE_NOISE', generation_input['packed_init_noises'])

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        t0 = time.time()
        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )
        if isinstance(unpacked_latent, (tuple, list)):
            # ``generate_image`` returns one latent slice per packed sequence;
            # the base inferencer decodes ``unpacked_latent[0]``.
            target_latent = unpacked_latent[0]
        else:
            target_latent = unpacked_latent
        _diagnose_tensor('GENERATED_IMAGE_LATENT', target_latent)

        t1 = time.time()
        image = self.decode_image(target_latent, image_shape)
        gpu_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available() else 0)
        print(f'[GEN_IMAGE] dt={t1 - t0:.1f}s shape={image_shape} '
              f'steps={num_timesteps} taylorseer={enable_taylorseer} gpu_peak={gpu_mb:.0f}MB',
              file=sys.stderr, flush=True)
        return image


    @torch.no_grad()
    def gen_text(
            self, gen_context, max_length: int = 500,
            do_sample: bool = True, temperature: float = 1.0):
        import sys
        import time as _time
        t0 = _time.time()
        ctx = deepcopy(gen_context)

        def make_generation_input():
            return self.model.prepare_start_tokens(
                ctx['kv_lens'], ctx['ropes'], self.new_token_ids)
        unpacked_latent = generate_text(
            self.model, make_generation_input, ctx['past_key_values'],
            max_length, do_sample, temperature,
            self.new_token_ids['eos_token_id'])
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        n_tokens = unpacked_latent.shape[0]  # dim=0 is tokens, dim=1 is batch
        dt = _time.time() - t0
        truncated = n_tokens >= max_length and '</think>' not in output
        print(f'[GEN_TEXT] dt={dt:.1f}s tokens={n_tokens}/{max_length} '
              f'trunc={truncated} sample={do_sample} t={temperature}',
              file=sys.stderr, flush=True)
        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: list[str | Image.Image],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=(0.4, 1.0),
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        image_seed=None,
    ) -> list[str | Image.Image]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        # Reference images update context, never the requested output geometry.
        step_image_shapes = image_shapes

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    orig_term = pil_img2rgb(input_term)
                    orig_size = orig_term.size
                    input_term = self.vae_transform.resize_transform(orig_term)
                    _diag(
                        f'INFER cond_img orig={orig_size} '
                        f'resized={input_term.size[::-1]}')
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=not understanding_output)

                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise TypeError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                # NaN guard: check KV cache before gen_image
                for ctx_name, ctx in [('gen', gen_context), ('text_cfg', cfg_text_context), ('img_cfg', cfg_img_context)]:
                    try:
                        pkv = ctx['past_key_values']
                        kv_tensors = pkv.key_cache if hasattr(pkv, 'key_cache') else []
                        if kv_tensors:
                            t_last = kv_tensors[-1] if kv_tensors else None
                            if t_last is not None:
                                has_nan = bool(torch.isnan(t_last).any().item())
                                has_inf = bool(torch.isinf(t_last).any().item())
                                if has_nan or has_inf:
                                    _diag(f'PRE_GEN NaN/Inf in {ctx_name} KV cache: NaN={has_nan} Inf={has_inf} '
                                          f'shape={t_last.shape} min={t_last.min().item():.4f} max={t_last.max().item():.4f}')
                    except (KeyError, AttributeError, RuntimeError, TypeError) as error:
                        _diag(
                            f'PRE_GEN diagnostic unavailable for {ctx_name}: '
                            f'{error}')
                _diag(f'PRE_GEN image_shapes={step_image_shapes} num_timesteps={num_timesteps}')
                img = self.gen_image(
                    step_image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                    image_seed=image_seed,
                )

                output_list.append(img)

        return output_list
