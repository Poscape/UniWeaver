"""Shared BAGEL text transport for UniWeaver trajectories."""

from core.text_protocol import (
    bagel_special_token_ids,
    encode_bagel_runtime_prefix,
    render_bagel_chat_prompt,
)


class BagelTextBackend:
    """Run role-correct text-only BAGEL inference for UniWeaver consumers."""

    def __init__(self, model, tokenizer, new_token_ids=None):
        self.model = model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids or self._token_ids(tokenizer)

    @staticmethod
    def _token_ids(tokenizer):
        return bagel_special_token_ids(tokenizer)

    def render_prompt(self, system_text, user_text):
        """Render one canonical system/user exchange with the tokenizer template."""
        return render_bagel_chat_prompt(
            self.tokenizer, system_text, user_text)

    def input_token_count(self, system_text, user_text):
        """Count the exact request tokens consumed by the BAGEL model."""
        return len(encode_bagel_runtime_prefix(
            self.tokenizer, system_text, user_text, self.new_token_ids))

    def output_token_count(self, response_text):
        """Count the response-content tokens returned to the caller."""
        return len(self.tokenizer.encode(response_text))

    @staticmethod
    def _device(model):
        return model.language_model.model.embed_tokens.weight.device

    @classmethod
    def _move_inputs(cls, model, generation_input):
        import torch

        device = cls._device(model)
        if device.type == 'meta':
            return generation_input
        for key, value in generation_input.items():
            if torch.is_tensor(value):
                generation_input[key] = value.to(device)
        return generation_input

    @staticmethod
    def _decode(tokenizer, token_ids):
        text = tokenizer.decode(token_ids)
        text = text.split('<|im_end|>')[0]
        if '<|im_start|>assistant' in text:
            return text.split('<|im_start|>assistant')[-1].strip()
        if '<|im_start|>' in text:
            return text.split('<|im_start|>')[-1].strip()
        return text.strip()

    def send(self, system_text, user_text, max_tokens=2048,
             do_sample=True, temperature=0.7):
        """Generate one response through the canonical batch decoder."""
        return self.send_many(
            system_text, [user_text], max_tokens=max_tokens,
            do_sample=do_sample, temperature=temperature)[0]

    def send_many(self, system_text, user_texts, max_tokens=2048,
                  do_sample=False, temperature=0.3):
        """Generate one or more responses through one canonical decoder."""
        import torch
        from torch.nn import functional

        from modeling.bagel.qwen2_navit import NaiveCache

        if not user_texts:
            return []
        if do_sample and temperature <= 0:
            raise ValueError('temperature must be positive when sampling')

        model = self.model
        tokenizer = self.tokenizer
        rendered = [
            self.render_prompt(system_text, user_text)
            for user_text in user_texts
        ]
        batch_size = len(rendered)
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        new_lengths = [0] * batch_size
        new_rope = [0] * batch_size
        generation_input, new_lengths, new_rope = model.prepare_prompts(
            curr_kvlens=new_lengths, curr_rope=new_rope, prompts=rendered,
            tokenizer=tokenizer, new_token_ids=self.new_token_ids)
        generation_input = self._move_inputs(model, generation_input)
        with torch.inference_mode(), torch.amp.autocast(
                'cuda', enabled=True, dtype=torch.bfloat16):
            past_key_values = model.forward_cache_update_text(
                past_key_values, **generation_input)

        generation_input = model.prepare_start_tokens(
            new_lengths, new_rope, self.new_token_ids)
        generation_input = self._move_inputs(model, generation_input)
        packed_key_value_indexes = generation_input['packed_key_value_indexes']
        key_values_lens = generation_input['key_values_lens']
        current_tokens = generation_input['packed_start_tokens']
        packed_query_position_ids = generation_input['packed_query_position_ids']
        device = current_tokens.device
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated_sequence = [current_tokens]
        eos_token_id = self.new_token_ids['eos_token_id']

        with torch.inference_mode(), torch.amp.autocast(
                'cuda', enabled=True, dtype=torch.bfloat16):
            for _ in range(max_tokens):
                packed_text_embedding = (
                    model.language_model.model.embed_tokens(current_tokens))
                query_lens = torch.ones_like(current_tokens)
                packed_query_indexes = (
                    torch.cumsum(key_values_lens, dim=0) +
                    torch.arange(batch_size, device=device,
                                 dtype=key_values_lens.dtype))

                unpacked_indexes = list(
                    packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
                for index in range(batch_size):
                    unpacked_indexes[index] += index
                packed_key_value_indexes = torch.cat(unpacked_indexes, dim=0)

                extra_inputs = {'mode': 'und'} if model.use_moe else {}
                output = model.language_model.forward_inference(
                    packed_query_sequence=packed_text_embedding,
                    query_lens=query_lens,
                    packed_query_position_ids=packed_query_position_ids,
                    packed_query_indexes=packed_query_indexes,
                    past_key_values=past_key_values,
                    key_values_lens=key_values_lens,
                    packed_key_value_indexes=packed_key_value_indexes,
                    update_past_key_values=True,
                    is_causal=True,
                    **extra_inputs)
                past_key_values = output.past_key_values
                logits = model.language_model.lm_head(
                    output.packed_query_sequence).float()
                safe_logits = torch.nan_to_num(
                    logits, nan=0.0, posinf=1e4, neginf=-1e4)
                if do_sample:
                    probabilities = functional.softmax(
                        safe_logits / temperature, dim=-1)
                    valid = (
                        torch.isfinite(probabilities).all(dim=-1) &
                        (probabilities.sum(dim=-1) > 0))
                    next_tokens = torch.argmax(safe_logits, dim=-1)
                    if valid.any():
                        next_tokens[valid] = torch.multinomial(
                            probabilities[valid], num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(safe_logits, dim=-1)
                next_tokens = torch.where(
                    finished, torch.full_like(next_tokens, eos_token_id), next_tokens)
                finished |= next_tokens.eq(eos_token_id)
                current_tokens = next_tokens
                generated_sequence.append(current_tokens)

                unpacked_indexes = list(
                    packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
                for index in range(batch_size):
                    unpacked_indexes[index] = torch.cat([
                        unpacked_indexes[index],
                        unpacked_indexes[index][-1:].add(1),
                    ])
                packed_key_value_indexes = torch.cat(unpacked_indexes, dim=0)
                key_values_lens = key_values_lens + 1
                packed_query_position_ids = packed_query_position_ids + 1
                if finished.all():
                    break

        token_matrix = torch.stack(generated_sequence, dim=0).cpu()
        return [
            self._decode(tokenizer, token_matrix[:, index].tolist())
            for index in range(batch_size)
        ]
