"""Text decoding used inside persistent multimodal BAGEL contexts."""
from copy import deepcopy


def generate_text(
        model, make_generation_input, past_key_values, max_tokens,
        do_sample, temperature, end_token_id):
    """Generate reviewer/thinking text from an existing multimodal KV cache."""
    generation_input = make_generation_input()
    cache = deepcopy(past_key_values) if do_sample else past_key_values
    tokens = model.generate_text(
        past_key_values=cache,
        max_length=max_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.3,
        end_token_id=end_token_id,
        **generation_input)
    return tokens
