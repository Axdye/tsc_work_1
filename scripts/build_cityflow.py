#!/usr/bin/env python3
"""Local portability build. Binding upgrade is recorded, not a canonical paper environment."""
import json
import shutil
import subprocess
import sys
from reproduce import ROOT, sha256


def main():
    source = ROOT / 'vendor/CityFlow'
    if not (source / 'CMakeLists.txt').is_file():
        raise SystemExit(f'Missing CityFlow source: {source}')
    for name, original in [('pybind11', 'pybind11-v2.10.4'), ('rapidjson', 'rapidjson')]:
        dep = ROOT / 'vendor' / original
        if not dep.is_dir():
            raise SystemExit(f'Missing build dependency {dep}; see docs/REPRODUCTION.md')
        target = source / 'extern' / name
        if not target.exists() or not any(target.iterdir()):
            shutil.copytree(dep, target, dirs_exist_ok=True)
    for required in ('extern/pybind11/CMakeLists.txt',
                     'extern/rapidjson/include/rapidjson/document.h',
                     'extern/milo/dtoa_milo.h'):
        if not (source / required).is_file():
            raise SystemExit(f'Missing offline build input: {source / required}; '
                             'upload complete source/dependency directories before building')
    build = ROOT / '.build/cityflow'
    subprocess.run(['cmake', '-S', str(source), '-B', str(build),
                    '-DPYTHON_EXECUTABLE=' + sys.executable,
                    '-DGIT_SUBMODULE=OFF',
                    '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
                    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(ROOT / '.build/python'),
                    '-DCMAKE_BUILD_TYPE=Release'], check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'cityflow', '-j', '4'], check=True)
    record = {'python': sys.executable, 'binding': 'upstream pybind11 v2.10.4 instead of original submodule',
              'traffic_dynamics_source_modified': False,
              'cmake': subprocess.check_output(['cmake', '--version'], text=True).splitlines()[0],
              'dependency_archives': {p.name: sha256(p) for p in (ROOT / 'vendor').glob('*.tar.gz')
                                      if p.name in ('CityFlow.tar.gz', 'rapidjson.tar.gz', 'pybind11-v2.10.4.tar.gz')}}
    (ROOT / '.build/build.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
