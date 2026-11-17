"""Direct-engine diagnostics; deliberately separate from official paper metrics."""
from collections import deque
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import uuid


def phase_movements(intersection):
    phases = intersection['trafficLight']['lightphases']
    if len(phases) < 5:
        raise ValueError('Diagnostic expects benchmark phase 0 clearance and green phases 1..4')
    result = {}
    for phase in range(1, 5):
        pairs = set()
        for link_id in phases[phase]['availableRoadLinks']:
            link = intersection['roadLinks'][link_id]
            if link['type'] == 'turn_right':
                continue
            by_in = {}
            for lane_link in link['laneLinks']:
                incoming = link['startRoad'] + '_' + str(lane_link['startLaneIndex'])
                outgoing = link['endRoad'] + '_' + str(lane_link['endLaneIndex'])
                by_in.setdefault(incoming, set()).add(outgoing)
            for incoming, outgoing in by_in.items():
                pairs.add((incoming, tuple(sorted(outgoing))))
        result[phase] = sorted(pairs)
    return result


def scores(movements, queues):
    # One incoming lane is counted once per movement; downstream queues averaged
    # over reachable lanes. This is a diagnostic MP variant, not an upstream clone.
    return {phase: sum(queues.get(i, 0) - sum(queues.get(o, 0) for o in outs) / len(outs)
                       for i, outs in pairs) for phase, pairs in movements.items()}


def clipped_flows(flows, seconds):
    result = []
    for flow in flows:
        if flow['startTime'] >= seconds:
            continue
        if flow['interval'] <= 0:
            raise ValueError('Flow interval must be positive')
        item = dict(flow)
        end = item.get('endTime', -1)
        item['endTime'] = seconds - 1 if end == -1 else min(end, seconds - 1)
        result.append(item)
    return result


def run(args, root):
    if args.seconds <= 0 or args.drain_seconds < 0 or not 0 <= args.yellow < args.interval or args.advice_age < 0:
        raise ValueError('Invalid physical horizon, clearance, decision interval, or advice age')
    # A local portability build is optional; external environments can install CityFlow.
    sys.path.insert(0, str(root / '.build/python'))
    import cityflow
    spec = {'jinan': ('Jinan', '3_4', 'anon_3_4_jinan_real.json'),
            'hangzhou': ('Hangzhou', '4_4', 'anon_4_4_hangzhou_real.json'),
            'newyork_28x7': ('NewYork', '28_7', 'anon_28_7_newyork_real_double.json')}[args.dataset]
    folder = root / 'vendor/LLMTSCS/data' / spec[0] / spec[1]
    road = folder / ('roadnet_' + spec[1] + '.json')
    flow = folder / (args.flow or spec[2])
    roadnet = json.loads(road.read_text())
    flows = clipped_flows(json.loads(flow.read_text()), args.seconds)
    intersections = {i['id']: phase_movements(i) for i in roadnet['intersections'] if not i['virtual']}
    out = root / 'outputs' / ('cityflow-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    out.mkdir(parents=True)
    local_flow = out / 'flow.json'
    local_flow.write_text(json.dumps(flows))
    rows = []
    for seed in args.seeds:
        for policy in args.policies:
            trial = out / f'{policy}-s{seed}'
            trial.mkdir()
            config = {'interval': 1, 'seed': seed, 'dir': '',
                      'roadnetFile': str(road), 'flowFile': str(local_flow),
                      'rlTrafficLight': True, 'laneChange': False, 'saveReplay': False}
            config_path = trial / 'config.json'
            config_path.write_text(json.dumps(config, indent=2))
            engine = cityflow.Engine(str(config_path), thread_num=1)
            current = {i: 1 for i in intersections}
            for i in intersections:
                engine.set_tl_phase(i, 1)
            history = deque(maxlen=args.advice_age + 1)
            incoming = set(x for m in intersections.values() for pairs in m.values() for x, _ in pairs)
            q_integral = 0
            remaining_at_horizon = None
            api_att_at_horizon = None
            started = time.perf_counter()
            with (trial / 'decisions.jsonl').open('w') as log:
                t = 0
                horizon = args.seconds + args.drain_seconds
                while t < horizon:
                    q = engine.get_lane_waiting_vehicle_count()
                    pressure = {i: scores(m, q) for i, m in intersections.items()}
                    history.append(pressure)
                    chosen_scores = history[0] if policy == 'stale-maxpressure' else pressure
                    target = {i: (1 + (t // args.interval) % 4 if policy == 'fixedtime'
                                  else max(chosen_scores[i], key=chosen_scores[i].get)) for i in intersections}
                    changed = {i for i in intersections if target[i] != current[i]}
                    log.write(json.dumps({'time': t, 'actions': target,
                                          'changed': sorted(changed),
                                          'advice_age_steps': min(args.advice_age, len(history) - 1)
                                          if policy == 'stale-maxpressure' else 0}) + '\n')
                    for k in range(min(args.interval, horizon - t)):
                        for i in intersections:
                            engine.set_tl_phase(i, 0 if i in changed and k < args.yellow else target[i])
                        engine.next_step()
                        t += 1
                        if t <= args.seconds:
                            q = engine.get_lane_waiting_vehicle_count()
                            q_integral += sum(q.get(lane, 0) for lane in incoming)
                        if t == args.seconds:
                            remaining_at_horizon = len(engine.get_vehicles(include_waiting=True))
                            api_att_at_horizon = engine.get_average_travel_time()
                    current = target
                    if t >= args.seconds and not engine.get_vehicles(include_waiting=True):
                        break
            remaining = len(engine.get_vehicles(include_waiting=True))
            row = {'policy': policy, 'seed': seed, 'demand_horizon_s': args.seconds,
                   'end_time_s': t, 'drained': remaining == 0, 'remaining': remaining,
                   'remaining_at_horizon': remaining_at_horizon,
                   'engine_att_at_horizon_s': api_att_at_horizon,
                   'engine_att_end_s': engine.get_average_travel_time(),
                   'all_vehicle_att_s': engine.get_average_travel_time() if remaining == 0 else None,
                   'network_mean_queued_vehicles': q_integral / args.seconds,
                   'wall_seconds': time.perf_counter() - started}
            rows.append(row)
            (trial / 'metrics.json').write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
            del engine
    with (out / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for policy in args.policies:
        subset = [r for r in rows if r['policy'] == policy]
        values = [r['all_vehicle_att_s'] for r in subset if r['all_vehicle_att_s'] is not None]
        summary[policy] = {'runs': len(subset), 'drained_runs': len(values),
                           'att_mean_s': statistics.mean(values) if values else None,
                           'att_sample_sd_s': statistics.stdev(values) if len(values) > 1 else None}
    manifest = {'kind': 'direct-engine diagnostic, NOT official paper reproduction',
                'args': vars(args), 'engine_file': cityflow.__file__,
                'roadnet_sha256': hashlib.sha256(road.read_bytes()).hexdigest(),
                'input_flow_sha256': hashlib.sha256(flow.read_bytes()).hexdigest(),
                'executed_flow_sha256': hashlib.sha256(local_flow.read_bytes()).hexdigest(),
                'protocol': '1s engine; 30s default decisions; clearance within interval only on phase change; '
                            'right turns as supplied; phases 1..4; no lane change; '
                            'stop inflow at demand horizon, then bounded drain; all-vehicle ATT only if drained.',
                'metric_note': 'CityFlow native ATT includes unfinished elapsed time until drained; '
                               'queue metric is vehicles across incoming controlled lanes, not per-intersection average.',
                'summary': summary}
    (out / 'summary.json').write_text(json.dumps(manifest, indent=2))
    print('Saved:', out)
