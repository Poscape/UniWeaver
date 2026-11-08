"""Canonical BAGEL text token protocol shared by inference and text SFT."""


def bagel_special_token_ids(tokenizer):
    """Resolve the four BAGEL control-token ids from one tokenizer."""
    token_ids = {
        'bos_token_id': tokenizer.convert_tokens_to_ids('<|im_start|>'),
        'eos_token_id': tokenizer.convert_tokens_to_ids('<|im_end|>'),
        'start_of_image': tokenizer.convert_tokens_to_ids('<|vision_start|>'),
        'end_of_image': tokenizer.convert_tokens_to_ids('<|vision_end|>'),
    }
    unknown_id = getattr(tokenizer, 'unk_token_id', None)
    invalid = [
        name for name, value in token_ids.items()
        if (not isinstance(value, int) or value < 0
            or (isinstance(unknown_id, int) and value == unknown_id))
    ]
    if invalid:
        raise ValueError(f'BAGEL tokenizer is missing control tokens: {invalid}')
    return token_ids


def render_bagel_chat_prompt(tokenizer, system_text, user_text):
    """Render the sole system/user Planner exchange."""
    return tokenizer.apply_chat_template(
        [
            {'role': 'system', 'content': system_text},
            {'role': 'user', 'content': user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_bagel_context_prompt(tokenizer, rendered_prompt, token_ids=None):
    """Encode exactly what ``Bagel.prepare_prompts`` places in the KV cache."""
    control = token_ids or bagel_special_token_ids(tokenizer)
    return [
        control['bos_token_id'],
        *tokenizer.encode(rendered_prompt),
        control['eos_token_id'],
    ]


def encode_bagel_request(tokenizer, system_text, user_text, token_ids=None):
    """Render and encode one complete BAGEL Planner request context."""
    rendered = render_bagel_chat_prompt(tokenizer, system_text, user_text)
    return encode_bagel_context_prompt(tokenizer, rendered, token_ids)


def encode_bagel_runtime_prefix(
        tokenizer, system_text, user_text, token_ids=None):
    """Encode every program-supplied token before the first model prediction."""
    control = token_ids or bagel_special_token_ids(tokenizer)
    return [
        *encode_bagel_request(tokenizer, system_text, user_text, control),
        control['bos_token_id'],
    ]


def encode_bagel_response_target(tokenizer, target_text, token_ids=None):
    """Encode only model-predicted response tokens, including the stop token."""
    control = token_ids or bagel_special_token_ids(tokenizer)
    return [*tokenizer.encode(target_text), control['eos_token_id']]


def encode_bagel_training_exchange(
        tokenizer, system_text, user_text, target_text, token_ids=None):
    """Return the exact runtime request followed by its generated target."""
    control = token_ids or bagel_special_token_ids(tokenizer)
    prefix_ids = encode_bagel_runtime_prefix(
        tokenizer, system_text, user_text, control)
    response_ids = encode_bagel_response_target(
        tokenizer, target_text, control)
    return prefix_ids + response_ids, len(prefix_ids)
