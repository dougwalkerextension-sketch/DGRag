"""Verify real DeepSeek generation with only the local Ollama socket permitted."""
import json
import os
import sys
import socket
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import app

chunk = next(c for c in app.STATE['chunks'] if 'ebony' in c['text'].lower() and 'clock' in c['text'].lower())
messages = [{'role': 'system', 'content': 'Answer briefly using only the excerpt. Cite [1].'},
            {'role': 'user', 'content': 'What effect does the clock have on the musicians?\n[1] ' + chunk['text']}]
original_connect = socket.socket.connect
def local_only(connection, address):
    if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1') or address[1] != 11434:
        raise AssertionError('External network access during local answer generation')
    return original_connect(connection, address)

with patch('socket.socket.connect', new=local_only):
    answer = app.generate_answer(messages)
assert answer.strip()
result = {'generation': app.generation_settings(), 'external_connections_blocked': True,
          'answer': answer, 'source_url': chunk['url']}
(app.DATA / 'local-generation-check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
