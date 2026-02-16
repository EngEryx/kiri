"""Nerve data collector — aggregates anomaly scores from all atoms."""

import sys
import json
import random
import math
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from kiri.core import Atom, StateLanguage


def score_atoms(pulse_atom=None, rhythm_atom=None, drift_atom=None,
                pulse_obs=None, rhythm_obs=None, drift_obs=None):
    """Score observations through their respective atoms. Returns Nerve observation."""
    now = datetime.now()

    def _score(atom, obs_dict):
        if atom is None or obs_dict is None:
            return 0.0
        tokens = atom.lang.encode_observation(obs_dict)
        if len(tokens) < 2:
            return 0.0
        n = min(atom.block_size, len(tokens) - 1)
        keys = [[] for _ in range(atom.n_layer)]
        vals = [[] for _ in range(atom.n_layer)]
        scores = []
        for pos in range(n):
            logits = atom.forward(tokens[pos], pos, keys, vals)
            probs = atom._softmax(logits)
            target = tokens[pos + 1]
            target_prob = probs[target].data
            scores.append(-math.log(max(target_prob, 1e-10)))
        return sum(scores) / len(scores) if scores else 0.0

    return {
        'P': max(0, min(20, _score(pulse_atom, pulse_obs))),
        'R': max(0, min(20, _score(rhythm_atom, rhythm_obs))),
        'D': max(0, min(20, _score(drift_atom, drift_obs))),
        'H': now.hour,
        'W': now.weekday(),
        'ts': now.isoformat()
    }


def log_feedback(nerve_obs, action, data_dir, source='human'):
    """Log a Nerve observation with feedback action and source tag."""
    nerve_obs['action'] = action
    nerve_obs['source'] = source
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    day = nerve_obs['ts'][:10]
    path = data_dir / f'nerve_{day}.jsonl'
    with open(path, 'a') as f:
        f.write(json.dumps(nerve_obs) + '\n')
    return str(path)


def generate_synthetic(days=7):
    """Generate synthetic Nerve data with simulated feedback."""
    observations = []
    start = datetime.now() - timedelta(days=days)

    random.seed(33)

    for day in range(days):
        for hour in range(8, 19):
            ts = start + timedelta(days=day, hours=hour)
            weekday = ts.weekday()

            # Simulate atom scores
            pulse_score = random.gauss(2.0, 1.5)
            rhythm_score = random.gauss(1.5, 1.0)
            drift_score = random.gauss(1.0, 0.8)

            # Occasional spikes
            if random.random() < 0.1:
                pulse_score = random.gauss(8.0, 2.0)
            if random.random() < 0.05:
                rhythm_score = random.gauss(6.0, 1.5)

            # Simulate user feedback
            max_score = max(pulse_score, rhythm_score, drift_score)
            if max_score > 6.0:
                action = 'alert'
            elif max_score > 4.0:
                action = random.choice(['alert', 'suppress', 'ok'])
            elif max_score > 2.5:
                action = random.choice(['ok', 'suppress'])
            else:
                action = 'ok'

            obs = {
                'P': max(0, min(20, pulse_score)),
                'R': max(0, min(20, rhythm_score)),
                'D': max(0, min(20, drift_score)),
                'H': hour,
                'W': weekday,
                'action': action,
                'ts': ts.isoformat()
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
        path = data_dir / f'nerve_{day}.jsonl'
        with open(path, mode) as f:
            for obs in day_obs:
                f.write(json.dumps(obs) + '\n')
        files_written.append(str(path))

    return files_written


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Nerve data collector')
    parser.add_argument('--dry-run', action='store_true', help='generate synthetic data')
    parser.add_argument('--days', type=int, default=7, help='days of synthetic data')
    parser.add_argument('--data-dir', default=None, help='output directory')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(__file__).resolve().parent.parent.parent / 'data')

    if args.dry_run:
        print(f"generating {args.days} days of synthetic nerve data...")
        obs = generate_synthetic(days=args.days)
        files = save_observations(obs, args.data_dir, overwrite=True)
        print(f"wrote {len(obs)} observations across {len(files)} files:")
        for f in files:
            print(f"  {f}")
    else:
        print("nerve requires running atoms to score. use --dry-run for synthetic data.")
