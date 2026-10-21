"""Run real offline embeddings against cached Poe passages and save verification metadata."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import app
import numpy as np

texts = [chunk['name'] + '\n' + chunk['text'] for chunk in app.STATE['chunks'][:4]]
if not texts:
    raise SystemExit('Run python app.py --prepare first')
# An accidental inference request would fail this check, even on a network-enabled host.
with patch('socket.socket.connect', side_effect=AssertionError('Network access during local embeddings')):
    vectors = np.asarray(app.embed(texts))
    repeated = np.asarray(app.embed([texts[0]]))[0]
assert vectors.shape == (4, 384), vectors.shape
assert np.isfinite(vectors).all()
assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
assert np.allclose(vectors[0], repeated, atol=1e-5)
result = {'embedding': app.embedding_settings(), 'passages': len(texts), 'dimensions': vectors.shape[1],
          'normalized': True, 'repeatable': True, 'network_disabled': True,
          'source_urls': list(app.SOURCES), 'input_characters': [len(text) for text in texts]}
(app.DATA / 'local-embedding-check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
