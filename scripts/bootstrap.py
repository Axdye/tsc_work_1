#!/usr/bin/env python3
"""Download missing official sources; keep existing checkouts and archive hashes."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request
from reproduce import ROOT, SOURCES, inventory


def extract(archive, destination):
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        top = members[0].name.split('/')[0]
        for member in members:
            if member.name.startswith('/') or '..' in Path(member.name).parts or member.issym() or member.islnk():
                raise ValueError('Unsafe source archive member: ' + member.name)
        tar.extractall(destination.parent)
    unpacked = destination.parent / top
    if unpacked != destination:
        unpacked.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=SOURCES, default=list(SOURCES))
    parser.add_argument('--build-deps', action='store_true', help='Also download portability build headers')
    args = parser.parse_args()
    (ROOT / 'vendor').mkdir(parents=True, exist_ok=True)
    failures = []
    if args.build_deps:
        deps = [('pybind11-v2.10.4', 'https://codeload.github.com/pybind/pybind11/tar.gz/refs/tags/v2.10.4'),
                ('rapidjson', 'https://codeload.github.com/cityflow-project/rapidjson/tar.gz/refs/heads/master')]
        for name, url in deps:
            dest = ROOT / 'vendor' / name
            if not dest.exists():
                archive = ROOT / 'vendor' / (name + '.tar.gz')
                urllib.request.urlretrieve(url, archive)
                extract(archive, dest)
    for name in args.only:
        slug, branch = SOURCES[name]
        dest = ROOT / 'vendor' / name
        if dest.exists():
            print('Keep existing:', dest)
            continue
        archive = dest.with_suffix('.tar.gz')
        url = f'https://codeload.github.com/{slug}/tar.gz/refs/heads/{branch}'
        try:
            print('Downloading:', url, flush=True)
            partial = archive.with_suffix('.partial')
            urllib.request.urlretrieve(url, partial)
            partial.rename(archive)
            extract(archive, dest)
        except Exception as exc:
            failures.append({'repo': name, 'error': str(exc)})
    (ROOT / 'sources.lock.json').write_text(json.dumps(inventory(), indent=2))
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))


if __name__ == '__main__':
    main()
