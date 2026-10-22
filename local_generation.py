"""DeepSeek R1 generation through the local Ollama service."""
import os
import re

import requests

DEFAULT_MODEL = 'deepseek-r1:8b'


def generation_settings():
    return {'provider': 'ollama-local', 'model': os.getenv('DEEPSEEK_MODEL') or DEFAULT_MODEL,
            'endpoint': 'http://127.0.0.1:11434', 'device': 'ollama-managed'}


def generation_ready():
    try:
        response = requests.get('http://127.0.0.1:11434/api/tags', timeout=(1, 2))
        response.raise_for_status()
        return any(model.get('name') == generation_settings()['model']
                   for model in response.json().get('models', []))
    except (requests.RequestException, ValueError):
        return False


def generate_answer(messages):
    # Preserve full current evidence and instructions; bound only earlier dialogue.
    retained = [messages[0], *[{**m, 'content': m['content'][:1000]} for m in messages[1:-1]], messages[-1]]
    model = generation_settings()['model']
    try:
        response = requests.post('http://127.0.0.1:11434/api/chat',
            json={'model': model, 'messages': retained, 'stream': False, 'think': True,
                  'options': {'temperature': 0.6, 'num_ctx': 16384,
                              'num_predict': max(256, min(8192, int(os.getenv('DEEPSEEK_MAX_NEW_TOKENS', '2048'))))}},
            timeout=(10, 300))
        if response.status_code == 404:
            raise ValueError('DeepSeek R1 model is missing. Run ollama pull ' + model + '.')
        if not response.ok:
            raise ValueError(f'Local DeepSeek R1 returned HTTP {response.status_code}. Check Ollama and model availability.')
        # Ollama separates reasoning from the final answer. Never display the reasoning field.
        text = response.json().get('message', {}).get('content', '').strip()
        # Also handle older Ollama versions that embed thinking tags in content.
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if '<think>' in text:
            text = text.split('<think>', 1)[0].strip()
        if not text:
            raise ValueError('DeepSeek R1 returned no final answer. Increase DEEPSEEK_MAX_NEW_TOKENS or try a shorter question.')
        return text
    except requests.Timeout as exc:
        raise ValueError('Local DeepSeek R1 timed out. Try a shorter question or check Ollama.') from exc
    except requests.ConnectionError as exc:
        raise ValueError('Cannot connect to local DeepSeek R1. Start Ollama on port 11434.') from exc
