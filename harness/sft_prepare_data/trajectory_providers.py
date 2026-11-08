"""API-provider transport adapters for UniWeaver trajectory construction."""
import os
import time

from core.production import API_THINKING_MODE

API_CONFIGS = {
    'deepseek': {'model': 'deepseek-v4-pro',
                 'label': 'deepseek', 'max_tok': 2048,
                 'thinking_control': 'deepseek_disabled',
                 'output_control': 'chat_json_object'},
}

class ChatSession:
    """One model request wrapper; the caller creates a fresh session per block."""
    def __init__(self, client, model, max_tok, thinking_control, output_control):
        self.client = client
        self.model = model
        self.max_tok = max_tok
        if thinking_control != 'deepseek_disabled':
            raise ValueError(
                f'UniWeaver has no verified thinking control for '
                f'{thinking_control!r}')
        self.thinking_control = thinking_control
        if output_control != 'chat_json_object':
            raise ValueError(
                f'UniWeaver has no verified JSON output control for '
                f'{output_control!r}')
        self.output_control = output_control
        self.messages = []

    def system(self, text):
        self.messages.append({'role': 'system', 'content': text})

    @staticmethod
    def _sampling_temperature(do_sample, temperature):
        if not isinstance(do_sample, bool):
            raise TypeError('do_sample must be bool')
        if do_sample and temperature <= 0:
            raise ValueError('temperature must be positive when sampling')
        return temperature if do_sample else 0

    def send(self, user_text, max_tokens=None, do_sample=True, temperature=0.7):
        effective_temperature = self._sampling_temperature(
            do_sample, temperature)
        self.messages.append({'role': 'user', 'content': user_text})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=max_tokens or self.max_tok,
            temperature=effective_temperature,
            extra_body={'thinking': {'type': API_THINKING_MODE}},
            response_format={'type': 'json_object'},
        )
        self.last_model = getattr(response, 'model', None) or self.model
        message = response.choices[0].message
        assistant_text = message.content or ''
        reasoning_text = getattr(message, 'reasoning_content', '') or ''
        if reasoning_text:
            raise RuntimeError(
                'API provider returned reasoning despite the disabled '
                'thinking contract')
        self.last_usage = response.usage
        if '<think>' in assistant_text.lower():
            raise RuntimeError(
                'API provider returned an explicit think block despite the '
                'JSON-only contract')
        self.messages.append({'role': 'assistant', 'content': assistant_text})
        return assistant_text


def create_client(api_provider, api_key_override=None, base_url_override=None):
    """Resolve one provider's credentials and construct its OpenAI-compatible client."""
    config = API_CONFIGS.get(api_provider)
    if not config:
        raise ValueError(f'unknown API: {api_provider}')
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            'openai package is required for API trajectory construction') from exc

    api_key = api_key_override or os.environ.get('UMM_API_KEY', '')
    if not api_key:
        raise ValueError(
            f'{api_provider} requires --key or UMM_API_KEY')

    base_url = base_url_override or os.environ.get('UMM_API_BASE_URL', '')
    if not base_url:
        raise ValueError(
            f'{api_provider} requires --base-url or UMM_API_BASE_URL')

    return config, OpenAI(api_key=api_key, base_url=base_url, timeout=300)


class ApiStageSender:
    """Shared stateless stage sender with canonical transport provenance."""

    def __init__(self, api_provider, model, api_key_override=None,
                 base_url_override=None):
        self.api_provider = api_provider
        self.model = model
        self.config, self.client = create_client(
            api_provider, api_key_override, base_url_override)
        self.last_metadata = {}
        self.totals = {
            'estimated_input_tokens': 0,
            'estimated_output_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0,
            'duration_seconds': 0.0,
        }

    def __call__(self, user_text, label='', max_tokens=2048,
                 system_text='', do_sample=True, temperature=0.7):
        session = ChatSession(
            self.client, self.model, max_tok=max_tokens,
            thinking_control=self.config['thinking_control'],
            output_control=self.config['output_control'])
        session.system(system_text)
        started = time.time()
        response = session.send(
            user_text, max_tokens=max_tokens,
            do_sample=do_sample, temperature=temperature)
        duration = time.time() - started
        usage = getattr(session, 'last_usage', None)
        input_tokens = (
            getattr(usage, 'input_tokens',
                    getattr(usage, 'prompt_tokens', 0)) or 0
            if usage else 0)
        output_tokens = (
            getattr(usage, 'output_tokens',
                    getattr(usage, 'completion_tokens', 0)) or 0
            if usage else 0)
        reasoning_tokens = (
            getattr(
                getattr(usage, 'completion_tokens_details', None),
                'reasoning_tokens', 0) or 0)
        self.last_metadata = {
            'provider': self.api_provider,
            'requested_model': self.model,
            'response_model': session.last_model,
            'thinking_mode': 'disabled',
            'duration_seconds': duration,
            'estimated_input_tokens': len(system_text + user_text) // 4,
            'estimated_output_tokens': len(response) // 4,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'reasoning_tokens': reasoning_tokens,
            'token_count_scope': 'provider_reported_usage',
        }
        for name in self.totals:
            self.totals[name] += self.last_metadata[name]
        if label:
            print(
                f'    [{duration:.1f}s] prompt≈'
                f'{self.last_metadata["estimated_input_tokens"]}t '
                f'output≈{self.last_metadata["estimated_output_tokens"]}t | '
                f'API: in={input_tokens} out={output_tokens} '
                f'reasoning={reasoning_tokens}',
                flush=True)
        return response
