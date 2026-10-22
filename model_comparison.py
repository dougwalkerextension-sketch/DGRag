"""Concurrent Ask Poe outputs with isolated model failures."""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def openai_config(root):
    # Read the optional key from .env on each turn; never return it to the browser.
    values = {}
    path = root / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return {'key': values.get('OPENAI_API_KEY', '').strip(),
            'model': values.get('OPENAI_CHAT_MODEL') or os.getenv('OPENAI_CHAT_MODEL', 'gpt-4.1-mini')}


def openai_answer(messages, settings):
    if not settings['key']:
        return 'not connected'
    try:
        response = requests.post(
            'https://api.openai.com/v1/responses',
            headers={'Authorization': 'Bearer ' + settings['key']},
            json={'model': settings['model'], 'instructions': messages[0]['content'],
                  'input': messages[1:], 'max_output_tokens': 512, 'store': False},
            timeout=(10, 90))
        if not response.ok:
            raise ValueError(f'OpenAI returned HTTP {response.status_code}. Check your key, model access, and quota.')
        data = response.json()
        text = '\n'.join(part['text'] for item in data.get('output', []) if item.get('type') == 'message'
                         for part in item.get('content', []) if part.get('type') == 'output_text').strip()
        if not text:
            raise ValueError('OpenAI returned no text. Try again or check model settings.')
        return text
    except requests.Timeout as exc:
        raise ValueError('OpenAI timed out. Try again.') from exc
    except requests.ConnectionError as exc:
        raise ValueError('Cannot connect to OpenAI. Check network access.') from exc


def mistral_settings():
    return {'provider': 'ollama-local', 'model': os.getenv('MISTRAL_MODEL') or 'mistral:7b'}


def mistral_answer(messages):
    # Ollama is loopback-only; this column never calls a hosted Mistral service.
    retained = [messages[0], *[{**m, 'content': m['content'][:1000]} for m in messages[1:-1]], messages[-1]]
    try:
        response = requests.post('http://127.0.0.1:11434/api/chat',
            json={'model': mistral_settings()['model'], 'messages': retained, 'stream': False,
                  'options': {'temperature': 0, 'num_predict': 256, 'num_ctx': 16384}}, timeout=(10, 300))
        if response.status_code == 404:
            raise ValueError('Mistral model is missing. Run ollama pull ' + mistral_settings()['model'] + '.')
        if not response.ok:
            raise ValueError(f'Local Mistral returned HTTP {response.status_code}. Check Ollama and model availability.')
        text = response.json().get('message', {}).get('content', '').strip()
        if not text:
            raise ValueError('Local Mistral returned no text. Try again.')
        return text
    except requests.Timeout as exc:
        raise ValueError('Local Mistral timed out. Check Ollama or try a shorter question.') from exc
    except requests.ConnectionError as exc:
        raise ValueError('Cannot connect to local Mistral. Start Ollama on port 11434.') from exc


def compare_models(messages, matches, local_generate, local_model, mistral_model,
                   settings, remote_messages=None, emit=None, mistral_messages=None):
    specs = [('local', local_model, lambda: local_generate(messages)),
             ('mistral', mistral_model, lambda: mistral_answer(mistral_messages or messages)),
             ('openai', settings['model'], lambda: openai_answer(remote_messages or messages, settings))]
    outputs = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run): (key, model) for key, model, run in specs}
        for future in as_completed(futures):
            key, model = futures[future]
            try:
                text = future.result()
                status = 'not_connected' if key == 'openai' and not settings['key'] else 'complete'
            except Exception as exc:
                # Unexpected exceptions and provider response bodies can contain secrets.
                text = str(exc) if isinstance(exc, ValueError) else f'{model} could not respond. Check its configuration and try again.'
                status = 'error'
            output = {'id': key, 'model': model, 'status': status, 'text': text}
            outputs[key] = output
            if emit:
                emit({'type': 'model', 'output': output})
    return [outputs[key] for key, _, _ in specs]
