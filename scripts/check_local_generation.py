"""Verify real answer generation with all outbound socket connections blocked."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['GENERATION_MAX_NEW_TOKENS'] = '64'
import app

chunk = next(c for c in app.STATE['chunks'] if 'ebony' in c['text'].lower() and 'clock' in c['text'].lower())
messages = [{'role': 'system', 'content': 'Answer briefly using only the excerpt. Cite [1].'},
            {'role': 'user', 'content': 'What effect does the clock have on the musicians?\n[1] ' + chunk['text']}]
with patch('socket.socket.connect', side_effect=AssertionError('Network access during local answer generation')):
    answer = app.generate_answer(messages)
assert answer.strip()
result = {'generation': app.generation_settings(), 'network_disabled': True,
          'answer': answer, 'source_url': chunk['url']}
(app.DATA / 'local-generation-check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
