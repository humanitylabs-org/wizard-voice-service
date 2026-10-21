"""Explicit owner-only batch profile selection; never restarts services."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile
import yaml
from config import ProviderConfigurationError, load_provider_config


def select(path: Path, profile: str, *, expected: str) -> dict:
    path = path.expanduser().resolve(strict=True)
    with open(str(path) + '.lock', 'a') as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        root = yaml.safe_load(path.read_text())
        stt = root['stt']
        previous = stt.get('active_profile', '')
        if previous != expected:
            raise ProviderConfigurationError('active profile changed; read status again')
        if profile not in stt.get('profiles', {}):
            raise ProviderConfigurationError('unknown profile')
        stt['active_profile'] = profile
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.profile-', delete=False) as f:
                temporary = Path(f.name)
                os.chmod(temporary, 0o600)
                yaml.safe_dump(root, f, sort_keys=False)
                f.flush()
                os.fsync(f.fileno())
            config = load_provider_config(temporary)
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    return {'previous': previous, 'selected': profile, 'provider': config.name,
            'model': config.model, 'privacy': config.privacy, 'fallback': None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    s = sub.add_parser('select')
    s.add_argument('profile')
    s.add_argument('--expected', required=True, help='current profile from status; use empty string for legacy')
    args = p.parse_args(argv)
    try:
        if args.command == 'select':
            result = select(args.config, args.profile, expected=args.expected)
        else:
            c = load_provider_config(args.config)
            result = {'selected': c.profile, 'provider': c.name, 'model': c.model,
                      'privacy': c.privacy, 'fallback': None}
        print(json.dumps(result))
        return 0
    except (OSError, KeyError, TypeError, yaml.YAMLError, ProviderConfigurationError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
