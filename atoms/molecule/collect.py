"""Molecule data collector — generates synthetic observation->action->explanation triples."""

import sys
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kiri.atoms.molecule.config import (
    PULSE_TOKENS, RHYTHM_TOKENS, DRIFT_TOKENS, SCORE_TOKENS, TEMPORAL_TOKENS,
    SCENARIOS, ACTIONS,
)


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _gen_pulse(scenario):
    """Generate pulse observation for a scenario."""
    if scenario == 'cpu_spike':
        return {'p.C': random.gauss(88, 8), 'p.M': random.gauss(65, 15),
                'p.D': random.gauss(50, 15), 'p.S': random.gauss(40, 20),
                'p.L': random.gauss(12, 4), 'p.N': 1}
    elif scenario == 'memory_pressure':
        return {'p.C': random.gauss(50, 15), 'p.M': random.gauss(90, 6),
                'p.D': random.gauss(50, 15), 'p.S': random.gauss(70, 15),
                'p.L': random.gauss(6, 3), 'p.N': 1}
    elif scenario == 'network_issue':
        return {'p.C': random.gauss(30, 15), 'p.M': random.gauss(45, 15),
                'p.D': random.gauss(50, 15), 'p.S': random.gauss(10, 10),
                'p.L': random.gauss(3, 2), 'p.N': 0}
    elif scenario == 'normal_busy':
        return {'p.C': random.gauss(55, 12), 'p.M': random.gauss(60, 12),
                'p.D': random.gauss(50, 10), 'p.S': random.gauss(20, 10),
                'p.L': random.gauss(5, 2), 'p.N': 1}
    elif scenario == 'suppress_routine':
        return {'p.C': random.gauss(45, 12), 'p.M': random.gauss(55, 12),
                'p.D': random.gauss(50, 10), 'p.S': random.gauss(15, 8),
                'p.L': random.gauss(4, 2), 'p.N': 1}
    elif scenario == 'retrain_novel':
        return {'p.C': random.gauss(40, 15), 'p.M': random.gauss(50, 15),
                'p.D': random.gauss(50, 15), 'p.S': random.gauss(20, 12),
                'p.L': random.gauss(5, 3), 'p.N': 1}
    else:
        return {'p.C': random.gauss(25, 12), 'p.M': random.gauss(40, 12),
                'p.D': random.gauss(45, 10), 'p.S': random.gauss(10, 8),
                'p.L': random.gauss(2, 1.5), 'p.N': 1}


def _gen_rhythm(scenario, hour):
    """Generate rhythm observation for a scenario."""
    if scenario == 'night_activity':
        return {'r.I': random.gauss(100, 60), 'r.A': random.gauss(35, 10),
                'r.H': hour, 'r.W': random.randint(0, 6)}
    elif scenario == 'normal_idle':
        return {'r.I': random.gauss(1200, 400), 'r.A': random.gauss(3, 2),
                'r.H': hour, 'r.W': random.randint(0, 4)}
    elif scenario in ('normal', 'normal_busy', 'suppress_known', 'suppress_routine'):
        return {'r.I': random.gauss(300, 200), 'r.A': random.gauss(15, 8),
                'r.H': hour, 'r.W': random.randint(0, 4)}
    else:
        return {'r.I': random.gauss(200, 150), 'r.A': random.gauss(20, 10),
                'r.H': hour, 'r.W': random.randint(0, 6)}


def _gen_drift(scenario):
    """Generate drift observation for a scenario."""
    if scenario == 'task_overload':
        return {'d.T': random.randint(6, 15), 'd.C': random.randint(0, 2),
                'd.S': random.randint(4, 9), 'd.H': random.randint(8, 18),
                'd.W': random.randint(0, 2)}
    elif scenario in ('retrain_signal', 'retrain_novel'):
        return {'d.T': random.randint(3, 8), 'd.C': random.randint(1, 4),
                'd.S': random.randint(2, 5), 'd.H': random.randint(8, 18),
                'd.W': random.randint(0, 4)}
    elif scenario == 'normal_idle':
        return {'d.T': random.randint(0, 1), 'd.C': random.randint(0, 1),
                'd.S': 0, 'd.H': random.randint(8, 18),
                'd.W': random.randint(0, 4)}
    else:
        return {'d.T': random.randint(1, 5), 'd.C': random.randint(1, 4),
                'd.S': random.randint(0, 2), 'd.H': random.randint(8, 18),
                'd.W': random.randint(0, 4)}


def _gen_scores(scenario):
    """Generate anomaly scores correlated with scenario."""
    if scenario == 'cpu_spike':
        return {'PS': random.gauss(8, 2), 'RS': random.gauss(1.5, 1), 'DS': random.gauss(1, 0.5)}
    elif scenario == 'memory_pressure':
        return {'PS': random.gauss(6, 2), 'RS': random.gauss(1.5, 1), 'DS': random.gauss(1, 0.5)}
    elif scenario == 'night_activity':
        return {'PS': random.gauss(2, 1), 'RS': random.gauss(7, 2), 'DS': random.gauss(1, 0.5)}
    elif scenario == 'task_overload':
        return {'PS': random.gauss(1.5, 1), 'RS': random.gauss(2, 1), 'DS': random.gauss(6, 2)}
    elif scenario == 'network_issue':
        return {'PS': random.gauss(5, 2), 'RS': random.gauss(1.5, 1), 'DS': random.gauss(1, 0.5)}
    elif scenario == 'retrain_signal':
        return {'PS': random.gauss(3, 1.5), 'RS': random.gauss(3, 1.5), 'DS': random.gauss(4, 1.5)}
    elif scenario == 'retrain_novel':
        return {'PS': random.gauss(4, 2), 'RS': random.gauss(4, 2), 'DS': random.gauss(3, 1.5)}
    elif scenario in ('suppress_known', 'suppress_routine'):
        return {'PS': random.gauss(4, 1.5), 'RS': random.gauss(1.5, 1), 'DS': random.gauss(1, 0.5)}
    elif scenario == 'normal_idle':
        return {'PS': random.gauss(0.5, 0.3), 'RS': random.gauss(0.5, 0.3), 'DS': random.gauss(0.3, 0.2)}
    elif scenario == 'normal_busy':
        return {'PS': random.gauss(1.8, 0.6), 'RS': random.gauss(1.2, 0.5), 'DS': random.gauss(0.8, 0.4)}
    else:
        return {'PS': random.gauss(1.2, 0.8), 'RS': random.gauss(1.0, 0.6), 'DS': random.gauss(0.8, 0.5)}


def generate_synthetic(n_samples=2000):
    """Generate synthetic molecule training data."""
    random.seed(42)

    # Build weighted scenario list
    scenarios = []
    for name, cfg in SCENARIOS.items():
        count = max(1, int(n_samples * cfg['weight']))
        scenarios.extend([name] * count)
    random.shuffle(scenarios)
    scenarios = scenarios[:n_samples]

    observations = []
    start = datetime.now() - timedelta(days=7)

    for i, scenario in enumerate(scenarios):
        ts = start + timedelta(seconds=i * 30)
        hour = ts.hour
        weekday = ts.weekday()

        # Night scenarios force nighttime hours
        if scenario == 'night_activity':
            hour = random.choice([0, 1, 2, 3, 4, 23])
        elif scenario == 'suppress_known':
            weekday = random.choice([5, 6])

        pulse = {k: _clamp(v, s[0], s[1])
                 for k, s in PULSE_TOKENS.items()
                 for kk, v in _gen_pulse(scenario).items() if kk == k}
        # Simpler: generate then clamp
        pulse = _gen_pulse(scenario)
        for k in pulse:
            if k in PULSE_TOKENS:
                lo, hi, _ = PULSE_TOKENS[k]
                pulse[k] = _clamp(pulse[k], lo, hi)

        rhythm = _gen_rhythm(scenario, hour)
        for k in rhythm:
            if k in RHYTHM_TOKENS:
                lo, hi, _ = RHYTHM_TOKENS[k]
                rhythm[k] = _clamp(rhythm[k], lo, hi)

        drift = _gen_drift(scenario)
        for k in drift:
            if k in DRIFT_TOKENS:
                lo, hi, _ = DRIFT_TOKENS[k]
                drift[k] = _clamp(drift[k], lo, hi)

        scores = _gen_scores(scenario)
        for k in scores:
            if k in SCORE_TOKENS:
                lo, hi, _ = SCORE_TOKENS[k]
                scores[k] = _clamp(scores[k], lo, hi)

        temporal = {'H': hour, 'W': weekday}

        cfg = SCENARIOS[scenario]
        explanation = random.choice(cfg['explanations'])

        obs = {
            'scenario': scenario,
            'action': cfg['action'],
            'explanation': explanation,
            'pulse': pulse,
            'rhythm': rhythm,
            'drift': drift,
            'scores': scores,
            'temporal': temporal,
            'ts': ts.isoformat(),
        }
        observations.append(obs)

    return observations


def save_observations(observations, data_dir, overwrite=False):
    """Save observations as JSONL, one file per day."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    by_day = {}
    for obs in observations:
        day = obs['ts'][:10]
        by_day.setdefault(day, []).append(obs)

    mode = 'w' if overwrite else 'a'
    files_written = []
    for day, day_obs in sorted(by_day.items()):
        path = data_dir / f'molecule_{day}.jsonl'
        with open(path, mode) as f:
            for obs in day_obs:
                f.write(json.dumps(obs) + '\n')
        files_written.append(str(path))

    return files_written


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Molecule data collector')
    parser.add_argument('--dry-run', action='store_true', help='generate synthetic data')
    parser.add_argument('--samples', type=int, default=2000, help='number of samples')
    parser.add_argument('--data-dir', default=None, help='output directory')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(__file__).resolve().parent.parent.parent / 'data')

    if args.dry_run:
        print(f"generating {args.samples} synthetic molecule samples...")
        obs = generate_synthetic(n_samples=args.samples)
        files = save_observations(obs, args.data_dir, overwrite=True)
        # Count scenario distribution
        from collections import Counter
        dist = Counter(o['scenario'] for o in obs)
        print(f"wrote {len(obs)} observations across {len(files)} files:")
        for f in files:
            print(f"  {f}")
        print(f"\nscenario distribution:")
        for name, count in dist.most_common():
            print(f"  {name:20s} {count:4d} ({100*count/len(obs):.0f}%)")
    else:
        print("molecule requires atom scores. use --dry-run for synthetic data.")
