"""Drift data collector — logs task activity for scope creep detection."""

import sys
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def log_entry(added=0, completed=0, switched=0):
    """Create a single task activity observation."""
    now = datetime.now()
    return {
        'T': max(0, min(20, added)),
        'C': max(0, min(20, completed)),
        'S': max(0, min(10, switched)),
        'H': now.hour,
        'W': now.weekday(),
        'ts': now.isoformat()
    }


def generate_synthetic(days=7, interval_minutes=60):
    """Generate synthetic task pattern data (1 entry per hour during work hours)."""
    observations = []
    start = datetime.now() - timedelta(days=days)
    steps = (days * 24 * 60) // interval_minutes

    random.seed(55)

    for i in range(steps):
        ts = start + timedelta(minutes=i * interval_minutes)
        hour = ts.hour
        weekday = ts.weekday()
        is_workday = weekday < 5

        if not (is_workday and 8 <= hour <= 18):
            continue

        if hour < 12:
            # Morning: focused, low additions, steady completions
            added = max(0, int(random.gauss(1, 1)))
            completed = max(0, int(random.gauss(2, 1)))
            switched = max(0, int(random.gauss(0.5, 0.5)))
        else:
            # Afternoon: more context switching
            added = max(0, int(random.gauss(2, 1.5)))
            completed = max(0, int(random.gauss(1, 1)))
            switched = max(0, int(random.gauss(1.5, 1)))

        # Occasional scope creep days (Wednesdays)
        if weekday == 2:
            added = max(0, int(random.gauss(4, 2)))
            switched = max(0, int(random.gauss(3, 1)))

        obs = {
            'T': max(0, min(20, added)),
            'C': max(0, min(20, completed)),
            'S': max(0, min(10, switched)),
            'H': hour,
            'W': weekday,
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
        path = data_dir / f'drift_{day}.jsonl'
        with open(path, mode) as f:
            for obs in day_obs:
                f.write(json.dumps(obs) + '\n')
        files_written.append(str(path))

    return files_written


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Drift data collector')
    parser.add_argument('--dry-run', action='store_true', help='generate synthetic data')
    parser.add_argument('--days', type=int, default=7, help='days of synthetic data (default: 7)')
    parser.add_argument('--added', type=int, default=0, help='tasks added this period')
    parser.add_argument('--completed', type=int, default=0, help='tasks completed this period')
    parser.add_argument('--switched', type=int, default=0, help='project switches this period')
    parser.add_argument('--data-dir', default=None, help='output directory')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(__file__).resolve().parent.parent.parent / 'data')

    if args.dry_run:
        print(f"generating {args.days} days of synthetic drift data...")
        obs = generate_synthetic(days=args.days)
        files = save_observations(obs, args.data_dir, overwrite=True)
        print(f"wrote {len(obs)} observations across {len(files)} files:")
        for f in files:
            print(f"  {f}")
    else:
        obs = log_entry(added=args.added, completed=args.completed, switched=args.switched)
        files = save_observations([obs], args.data_dir)
        print(json.dumps(obs, indent=2))
        print(f"saved -> {files[0]}")
