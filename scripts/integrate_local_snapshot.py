"""Three-way integration into the isolated upgrade checkout; preserve conflicts separately."""
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(r'D:\chatgpt\aiagents-stock\tick-stock-panel-v022-integrated')
TARGET = Path(__file__).resolve().parents[1]
REPORT = Path(r'F:\tick-panel-upgrade-20261009\integration')
REPORT.mkdir(parents=True, exist_ok=True)

def git(*args):
    return subprocess.run(['git', '-c', 'core.quotepath=false', *args], cwd=SOURCE,
                          capture_output=True, check=True).stdout

changed = git('diff', '--name-only', 'HEAD', '-z').decode().split('\0')
untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
result = {'merged': [], 'copied': [], 'conflicts': [], 'skipped': []}
for name in sorted(set(changed + untracked) - {''}):
    if name.startswith('tmp/'):
        result['skipped'].append(name)
        continue
    local, dest = SOURCE / name, TARGET / name
    if not local.is_file():
        result['skipped'].append(name)
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(local, dest)
        result['copied'].append(name)
        continue
    try:
        base = git('show', 'HEAD:' + name)
    except subprocess.CalledProcessError:
        base = b''
    if local.read_bytes().replace(b'\r\n', b'\n') == dest.read_bytes().replace(b'\r\n', b'\n'):
        continue
    basepath = REPORT / 'base.tmp'
    basepath.write_bytes(base.replace(b'\r\n', b'\n'))
    localpath = REPORT / 'local.tmp'
    uppath = REPORT / 'upstream.tmp'
    localpath.write_bytes(local.read_bytes().replace(b'\r\n', b'\n'))
    uppath.write_bytes(dest.read_bytes().replace(b'\r\n', b'\n'))
    merged = subprocess.run(['git', 'merge-file', '-p', str(localpath), str(basepath), str(uppath)],
                            capture_output=True)
    if merged.returncode == 0:
        dest.write_bytes(merged.stdout)
        result['merged'].append(name)
    else:
        conflict = REPORT / 'conflicts' / name
        conflict.parent.mkdir(parents=True, exist_ok=True)
        conflict.write_bytes(merged.stdout)
        result['conflicts'].append(name)
(REPORT / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print({key: len(value) for key, value in result.items()})
