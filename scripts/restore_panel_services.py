"""Restore paused Ubuntu services, waiting for app startup before shadow readers."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

ROLES = ('app', 'frozen-pair-shadow', 'decision-pair-shadow', 'repaired-pair-shadow')


def restore(roles, *, start, healthy, wait, monotonic, timeout=180):
    """Do not start readers until the app has finished its startup lifecycle."""
    if not roles or len(roles) != len(set(roles)) or set(roles) - set(ROLES):
        raise ValueError('Invalid paused service roles')
    if timeout <= 0:
        raise ValueError('Positive startup timeout required')
    if 'app' in roles:
        start('app')
    deadline = monotonic() + timeout
    while not healthy():
        if monotonic() >= deadline:
            raise TimeoutError('App startup not healthy; shadow services remain paused')
        wait(2)
    for role in ROLES[1:]:
        if role in roles:
            start(role)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maintenance-record', type=Path, required=True)
    parser.add_argument('--roles', nargs='+', choices=ROLES, required=True,
                        help='Only roles actually paused by this maintenance operation')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    record = json.loads(args.maintenance_record.read_text())
    images = record['images']
    if len(args.roles) != len(set(args.roles)) or any(r not in images for r in args.roles):
        raise ValueError('Paused roles do not match the saved maintenance record')
    for role in args.roles:
        result = subprocess.check_output(
            ['docker', 'inspect', f'tickpanel-v035-{role}-1'], text=True, timeout=15)
        if json.loads(result)[0]['Image'] != images[role]:
            raise ValueError('Image changed since maintenance began: ' + role)

    def healthy():
        try:
            with urlopen('http://127.0.0.1:3018/health', timeout=3) as stream:
                return json.load(stream).get('status') == 'ok'
        except (OSError, ValueError):
            return False

    if args.check_only:
        if not healthy():
            raise RuntimeError('App has not finished startup')
        print(json.dumps({'status': 'ready', 'services_started': 0, 'images_match': True}))
        return

    def start(role):
        subprocess.run(['docker', 'start', f'tickpanel-v035-{role}-1'], check=True, timeout=45)

    restore(args.roles, start=start, healthy=healthy, wait=time.sleep,
            monotonic=time.monotonic, timeout=args.timeout)
    print(json.dumps({'status': 'restored', 'roles': args.roles, 'app_ready_before_shadows': True}))


if __name__ == '__main__':
    main()
