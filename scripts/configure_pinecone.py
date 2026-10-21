"""Create/select the vector index required by local MiniLM, preserving existing indexes."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app

app.require_keys()
headers = {'Api-Key': app.os.environ['PINECONE_API_KEY'], 'X-Pinecone-Api-Version': '2025-10'}
base = 'https://api.pinecone.io/indexes'
response = app.requests.get(base, headers=headers, timeout=30)
response.raise_for_status()
indexes = response.json().get('indexes', [])
name = 'dgrag-poe-local'
index = next((item for item in indexes if item['name'] == name), None)
if index is None:
    old_host = app.os.environ['PINECONE_INDEX_HOST'].removeprefix('https://').rstrip('/')
    old = next((item for item in indexes if item.get('host') == old_host), {})
    serverless = old.get('spec', {}).get('serverless', {})
    spec = {key: serverless[key] for key in ('cloud', 'region') if key in serverless}
    if len(spec) != 2:
        spec = {'cloud': 'aws', 'region': 'us-east-1'}
    response = app.requests.post(base, headers=headers, json={
        'name': name, 'dimension': 384, 'metric': 'cosine', 'vector_type': 'dense',
        'spec': {'serverless': spec}}, timeout=30)
    if response.status_code not in (200, 201, 202):
        raise SystemExit(f'Index creation failed: HTTP {response.status_code}; no configuration changed.')
    print('Created dgrag-poe-local: 384 dimensions, cosine, local vectors', flush=True)
for attempt in range(30):
    response = app.requests.get(base + '/' + name, headers=headers, timeout=30)
    response.raise_for_status()
    index = response.json()
    if index.get('status', {}).get('ready'):
        break
    time.sleep(2)
else:
    raise SystemExit('Index is still initializing; rerun to complete configuration.')
if index.get('dimension') != 384 or index.get('metric') != 'cosine':
    raise SystemExit('Existing index has incompatible settings; no configuration changed.')
env = app.ROOT / '.env'
lines = env.read_text(encoding='utf-8').splitlines()
lines = [line for line in lines if not line.strip().startswith('PINECONE_INDEX_HOST=')]
lines.append('PINECONE_INDEX_HOST=' + index['host'])
env.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Updated .env to use dgrag-poe-local. Existing indexes were preserved.', flush=True)
