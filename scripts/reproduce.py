#!/usr/bin/env python3
"""Auditable entry point for isolated upstream runs and CPU CityFlow diagnostics."""
import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
METHODS = json.loads((ROOT / 'configs/methods.json').read_text())
SOURCES = {
    'LLMTSCS': ('usail-hkust/LLMTSCS', 'main'),
    'CoLLMLight': ('usail-hkust/CoLLMLight', 'master'),
    'HeraldLight': ('BUPT-ANTlab/HeraldLight', 'main'),
    'LLM-Assisted-Light': ('Traffic-Alpha/LLM-Assisted-Light', 'main'),
    'CityFlow': ('cityflow-project/CityFlow', 'master'),
    'LibSignal': ('DaRL-LibSignal/LibSignal', 'master'),
    'GENEX-TSC': ('Run-Aground/GENEX-TSC', 'master'),
    'VLALight': ('usail-hkust/VLALight', 'main'),
    'TransSimHub': ('Traffic-Alpha/TransSimHub', 'main'),
    'ProcessLight': ('wenzhaoabc/processlight', 'main'),
    'DynamicLight': ('LiangZhang1996/DynamicLight', 'main'),
    'X-Light': ('jiang-haoyuan/X-Light', 'main'),
    'Unicorn': ('marmotlab/Unicorn', 'main'),
    'DataLight': ('LiangZhang1996/DataLight', 'main'),
    'iLLM-TSC': ('Traffic-Alpha/iLLM-TSC', 'main'),
    'PromptGAT': ('DaRL-LibSignal/PromptGAT', 'main'),
    'VLMLight': ('Traffic-Alpha/VLMLight', 'main'),
    'RobustLight': ('MJLee00/RobustLight', 'main'),
    'CFLight': ('MJLee00/CFLight', 'main'),
    'DGLight': ('yyccbb/FYP_LLMTSC', 'main'),
}
FLOWS = {
    'jinan': 'anon_3_4_jinan_real.json',
    'hangzhou': 'anon_4_4_hangzhou_real.json',
    'newyork_28x7': 'anon_28_7_newyork_real_double.json',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    result = {}
    for name, (slug, branch) in SOURCES.items():
        repo = ROOT / 'vendor' / name
        item = {'url': 'https://github.com/' + slug, 'requested_branch': branch,
                'present': repo.is_dir()}
        if (repo / '.git').is_dir():
            item['git_commit'] = subprocess.check_output(
                ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            item['git_status'] = subprocess.check_output(
                ['git', '-C', str(repo), 'status', '--porcelain'], text=True).splitlines()
        archive = ROOT / 'vendor' / (name + '.tar.gz')
        if archive.exists():
            item['archive_sha256'] = sha256(archive)
            item['archive_url'] = f'https://codeload.github.com/{slug}/tar.gz/refs/heads/{branch}'
            item['version_note'] = 'Archive hash identifies downloaded bytes; mutable branch is not a verified commit pin.'
        if repo.is_dir():
            item['license_files'] = [p.name for p in repo.iterdir()
                                     if p.name.lower().startswith(('license', 'copying'))]
            # Hash Python/config sources, excluding build outputs and model weights.
            files = sorted(p for p in repo.rglob('*') if p.is_file()
                           and '.git' not in p.parts and p.suffix in ('.py', '.yaml', '.toml'))
            item['source_hashes'] = {str(p.relative_to(repo)): sha256(p) for p in files}
        result[name] = item
    return result


def doctor():
    sys.path.insert(0, str(ROOT / '.build/python'))
    report = {'python': sys.executable, 'version': sys.version,
              'modules': {m: importlib.util.find_spec(m) is not None
                          for m in ['cityflow', 'numpy', 'torch', 'tensorflow', 'vllm', 'openai', 'tshub']},
              'methods': {}}
    for name, method in METHODS.items():
        entry = ROOT / 'vendor' / (method.get('repo') or '') / method.get('entry', '')
        item = {'stage': method.get('stage'), 'blocked': method.get('blocked'),
                'entry_exists': entry.is_file(), 'note': method.get('note')}
        if entry.is_file():
            try:
                ast.parse(entry.read_text())
                item['syntax'] = 'ok (not an import/runtime test)'
            except SyntaxError as exc:
                item['syntax'] = str(exc)
        report['methods'][name] = item
    return report


def build_native_plan(args, out):
    method = METHODS[args.method]
    if method.get('blocked'):
        raise ValueError(method['blocked'])
    repo = ROOT / 'vendor' / method['repo']
    entry = repo / method['entry']
    if not entry.is_file():
        raise ValueError(f'Missing upstream entry: {entry}')
    command = [args.python, str(entry)]
    if method['family'] == 'cityflow-native':
        command += ['--dataset', args.dataset, '--traffic_file', args.flow or FLOWS[args.dataset]]
    elif method['family'] == 'tshub':
        event = repo / 'scenarios' / args.scenario / 'events' / args.event
        if not event.is_file():
            raise ValueError(f'Missing event file: {event}')
        command += ['--scenario', args.scenario, '--event-config', str(event),
                    '--decision-log', str(out / 'decisions.jsonl'),
                    '--trip-info', str(out / 'tripinfo.xml')]
        if args.method == 'la-light':
            command += ['--episode-summary', str(out / 'episode.json')]
    command += args.extra
    return {'method': args.method, 'stage': method.get('stage'), 'cwd': str(repo),
            'command': command, 'note': method.get('note'),
            'seed_status': 'Native defaults retained. No claim of independently controlled simulator/learner seeds.',
            'entry_sha256': sha256(entry), 'validation': 'not_executed'}


def native(args):
    out = ROOT / 'outputs' / (args.method + '-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    plan = build_native_plan(args, out)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print('Command:', shlex.join(plan['command']))
    if args.dry_run:
        return
    out.mkdir(parents=True)
    (out / 'sources.json').write_text(json.dumps(inventory(), indent=2))
    record = out / 'run.json'
    record.write_text(json.dumps(plan, indent=2))
    env = dict(os.environ, WANDB_MODE='disabled')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'vendor/TransSimHub'), env.get('PYTHONPATH', '')])
    if METHODS[args.method]['family'] == 'tshub' and not env.get('SUMO_HOME'):
        probe = subprocess.run([args.python, '-c',
            'import importlib.util; s=importlib.util.find_spec("sumo"); '
            'print(next(iter(s.submodule_search_locations)) if s else "")'],
            capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip():
            env['SUMO_HOME'] = probe.stdout.strip()
            env['PATH'] = os.pathsep.join([str(Path(args.python).resolve().parent),
                                         str(Path(env['SUMO_HOME']) / 'bin'), env.get('PATH', '')])
    try:
        with (out / 'stdout.log').open('w') as log:
            process = subprocess.Popen(plan['command'], cwd=plan['cwd'], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
            code = process.wait()
        plan.update(returncode=code, validation='process_exited_not_metric_reproduction')
    except Exception as exc:
        plan.update(validation='launch_failed', error=str(exc))
        code = 1
    record.write_text(json.dumps(plan, indent=2))
    print(f'Run record: {record}')
    if code:
        raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    sub.add_parser('doctor')
    sub.add_parser('lock')
    run = sub.add_parser('native', help='Run one official entry point in its own Python environment')
    run.add_argument('--method', choices=METHODS, required=True)
    run.add_argument('--python', default=sys.executable)
    run.add_argument('--dataset', choices=FLOWS, default='jinan')
    run.add_argument('--flow')
    run.add_argument('--scenario', choices=['3way', '4way'], default='4way')
    run.add_argument('--event', default='special_vehicles_accident.yaml')
    run.add_argument('--dry-run', action='store_true')
    smoke = sub.add_parser('cityflow', help='CPU diagnostic, not a faithful paper reproduction')
    smoke.add_argument('--dataset', choices=FLOWS, default='jinan')
    smoke.add_argument('--flow')
    smoke.add_argument('--policies', nargs='+', choices=['fixedtime', 'maxpressure', 'stale-maxpressure'],
                       default=['fixedtime', 'maxpressure'])
    smoke.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    smoke.add_argument('--seconds', type=int, default=3600)
    smoke.add_argument('--drain-seconds', type=int, default=3600)
    smoke.add_argument('--interval', type=int, default=30)
    smoke.add_argument('--yellow', type=int, default=5)
    smoke.add_argument('--advice-age', type=int, default=2, help='Decisions; stale policy is only a proxy diagnostic')
    args, extra = parser.parse_known_args()
    if extra and extra[0] == '--':
        extra = extra[1:]
    if args.cmd != 'native' and extra:
        parser.error('Unknown arguments: ' + ' '.join(extra))
    args.extra = extra
    try:
        if args.cmd == 'list':
            print(json.dumps(METHODS, indent=2, ensure_ascii=False))
        elif args.cmd == 'doctor':
            print(json.dumps(doctor(), indent=2, ensure_ascii=False))
        elif args.cmd == 'lock':
            path = ROOT / 'sources.lock.json'
            path.write_text(json.dumps(inventory(), indent=2))
            print(path)
        elif args.cmd == 'native':
            native(args)
        else:
            import cityflow_diagnostic
            cityflow_diagnostic.run(args, ROOT)
    except (ValueError, ImportError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
