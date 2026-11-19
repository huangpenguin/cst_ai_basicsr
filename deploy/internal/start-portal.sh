#!/usr/bin/env bash
# Mount checks precede all directory creation. Never materialize a missing NFS.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
portal_config=/home/huang/.config/basicsr-xray-portal
portal_state=/home/huang/.local/state/basicsr-xray-portal
portal_cloud=/home/huang/.local/state/basicsr-xray-portal-cloud
portal_data=/mnt/data/cst_ai/intake
if [[ $# -gt 1 || ( $# == 1 && "$1" != --check ) ]]; then
  echo 'Usage: bash deploy/internal/start-portal.sh [--check]' >&2
  exit 2
fi
[[ "$(id -u)" == 1003 ]] || { echo 'Run as work1 user huang (UID 1003).' >&2; exit 1; }
python3 - "$repo_dir" <<'PY'
import json, os, subprocess, sys, tempfile
from pathlib import Path

def mounted(path, expected_source):
    # stat activates autofs first; findmnt alone can otherwise report /mnt autofs.
    Path(path).stat()
    info=json.loads(subprocess.check_output(['findmnt','--json','--target',path,'--output','TARGET,SOURCE,FSTYPE,OPTIONS']))['filesystems'][0]
    if info['fstype'] not in ('nfs','nfs4') or info['source'] != expected_source or 'rw' not in info['options'].split(','):
        raise SystemExit(f'Refusing startup: {path} is not the expected writable NFS mount.')
    fd, name=tempfile.mkstemp(prefix='.portal-writecheck-',dir=path)
    try:
        os.write(fd,b'portal mount write check\n'); os.fsync(fd)
    finally:
        os.close(fd); os.unlink(name)
    print(f"Verified {path}: {info['source']} ({info['fstype']}, writable)")
mounted(sys.argv[1], '192.168.3.14:/mnt/storage/home')
mounted('/mnt/data/cst_ai','192.168.3.14:/mnt/storage/data')
for path in ['/home/huang/.local/state', '/home/huang/.config']:
    info=json.loads(subprocess.check_output(['findmnt','--json','--target',path,'--output','FSTYPE']))['filesystems'][0]
    if info['fstype'] in ('nfs','nfs4','autofs','cifs'):
        raise SystemExit('Private portal state/config must be on work1 local disk.')
if '192.168.3.11' not in subprocess.check_output(['hostname','-I'],text=True).split():
    raise SystemExit('Expected work1 LAN address 192.168.3.11 is absent.')
adc=Path('/home/huang/.config/basicsr-vertex/application_default_credentials.json')
if not adc.is_file() or adc.stat().st_mode & 0o077 or adc.stat().st_uid != os.getuid():
    raise SystemExit('Existing local user ADC must be owned by huang with private permissions.')
try:
    adc_type=json.loads(adc.read_text()).get('type')
except (OSError, ValueError):
    raise SystemExit('Existing local ADC is not readable valid JSON.') from None
if adc_type != 'authorized_user':
    raise SystemExit('The portal requires the existing authorized_user ADC.')
print('Verified local state/config filesystem, LAN IP and private authorized_user ADC (secrets not printed).')
PY
docker info --format '{{.OSType}}' >/dev/null
docker compose version >/dev/null
if [[ "${1:-}" == --check ]]; then
  if [[ -f "$portal_config/portal.env" ]]; then
    docker compose -f "$repo_dir/deploy/internal/compose.yaml" config --quiet
  fi
  echo 'Portal preflight passed. No services were started.'
  exit 0
fi
umask 077
mkdir -p "$portal_config" "$portal_state/tus" "$portal_cloud" "$portal_data"
chmod 700 "$portal_config" "$portal_state" "$portal_state/tus" "$portal_cloud"
python3 - "$portal_config" <<'PY'
import os, secrets, sys
from pathlib import Path
folder=Path(sys.argv[1]); config=folder/'portal.env'
if not config.exists():
    values={key:secrets.token_urlsafe(32) for key in ['PORTAL_ADMIN_PASSWORD','PORTAL_SESSION_SECRET','PORTAL_HOOK_SECRET']}
    with config.open('x') as output:
        output.write(''.join(f'{key}={value}\n' for key,value in values.items()))
    os.chmod(config,0o600)
    password=folder/'admin-password.txt'
    with password.open('x') as output: output.write(values['PORTAL_ADMIN_PASSWORD']+'\n')
    os.chmod(password,0o600)
PY
docker compose -f "$repo_dir/deploy/internal/compose.yaml" config --quiet
docker compose -f "$repo_dir/deploy/internal/compose.yaml" build api
docker compose -f "$repo_dir/deploy/internal/compose.yaml" up -d --no-build
echo 'Portal started: http://192.168.3.11:7862'
echo 'Admin password is in /home/huang/.config/basicsr-xray-portal/admin-password.txt (keep it private).'
