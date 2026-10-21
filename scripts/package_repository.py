"""Prepare a source-only GitHub upload without local credentials, models, or runtime data."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
names = ['.gitignore', '.env.example', 'app.py', 'local_generation.py', 'index.html',
         'ask-poe.js', 'README.md', 'WORK_SUMMARY.md', 'requirements.txt', 'start.ps1']
files = [root / name for name in names]
files += sorted((root / 'scripts').glob('*.py'))
files += sorted((root / 'tests').glob('*.py'))
secrets = []
env = root / '.env'
if env.exists():
    for line in env.read_text(encoding='utf-8').splitlines():
        if '=' not in line or line.lstrip().startswith('#'):
            continue
        key, value = line.split('=', 1)
        value = value.strip().strip('"').strip("'")
        if any(part in key.upper() for part in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET')) and len(value) >= 12:
            secrets.append(value)
for path in files:
    content = path.read_text(encoding='utf-8')
    if any(secret in content for secret in secrets):
        raise SystemExit('Configured secret detected in ' + str(path.relative_to(root)) + '; packaging stopped.')
destination = root / 'dist'
destination.mkdir(exist_ok=True)
archive = destination / 'DGRag-source.zip'
with ZipFile(archive, 'w', ZIP_DEFLATED) as output:
    for path in files:
        output.write(path, path.relative_to(root).as_posix())
manifest = {'files': [str(path.relative_to(root)).replace('\\', '/') for path in files],
            'excluded': ['.env and local env variants', 'data including credentials-adjacent artifacts and model weights',
                         '.runtime', '.venv', 'python-runtime.zip', '__pycache__', 'dist'],
            'configured_secret_scan': 'passed', 'archive_bytes': archive.stat().st_size}
(destination / 'publish-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
