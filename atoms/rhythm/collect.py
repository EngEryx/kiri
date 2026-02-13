"""Rhythm data collector — tracks work patterns via keyboard/mouse idle time."""

import sys
import json
import random
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path


def _get_idle_seconds():
    """Get HID idle time in seconds on macOS."""
    try:
        r = subprocess.run(
            ['ioreg', '-c', 'IOHIDSystem'],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            return None
        m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', r.stdout)
        if m:
            return int(m.group(1)) / 1_000_000_000  # nanoseconds to seconds
        return None
    except Exception:
        return None


class ActivityTracker:
    """Tracks active/idle transitions and derives session metrics."""

    def __init__(self, idle_threshold=300):
        self.idle_threshold = idle_threshold  # seconds before considered idle
        self.last_idle = None
        self.last_active_ts = None
        self.events_in_window = 0
        self.window_start = datetime.now()

    def sample(self):
        """Take a sample. Returns observation dict or None if not enough data."""
        idle_secs = _get_idle_seconds()
        if idle_secs is None:
            return None

        now = datetime.now()
        window_duration = (now - self.window_start).total_seconds()

        # Activity density: transitions from idle to active per minute
        is_active = idle_secs < self.idle_threshold
        if self.last_idle is not None:
            was_idle = self.last_idle >= self.idle_threshold
            if is_active and was_idle:
                self.events_in_window += 1

        self.last_idle = idle_secs

        # Calculate density over the window
        density = 0
        if window_duration > 0:
            density = (self.events_in_window / window_duration) * 60  # events/min

        # Reset window every 5 minutes
        if window_duration >= 300:
            self.events_in_window = 0
            self.window_start = now

        return {
            'I': max(0, min(3600, idle_secs)),
            'A': max(0, min(60, density)),
            'H': now.hour,
            'W': now.weekday(),
            'ts': now.isoformat()
        }


def generate_synthetic(days=7, interval_minutes=5):
    """Generate synthetic work pattern data."""
    observations = []
    start = datetime.now() - timedelta(days=days)
    steps = (days * 24 * 60) // interval_minutes

    random.seed(77)

    for i in range(steps):
        ts = start + timedelta(minutes=i * interval_minutes)
        hour = ts.hour
        weekday = ts.weekday()
        is_workday = weekday < 5

        if is_workday and 9 <= hour <= 12:
            # Morning focus: low idle, high activity
            idle = random.gauss(30, 20)
            activity = random.gauss(8, 3)
        elif is_workday and 13 <= hour <= 17:
            # Afternoon: moderate idle, moderate activity
            idle = random.gauss(120, 60)
            activity = random.gauss(5, 2)
        elif is_workday and 20 <= hour <= 23:
            # Evening side work
            idle = random.gauss(300, 150)
            activity = random.gauss(2, 1)
        else:
            # Off hours / weekends: high idle
            idle = random.gauss(1800, 600)
            activity = random.gauss(0.5, 0.3)

        obs = {
            'I': max(0, min(3600, idle)),
            'A': max(0, min(60, activity)),
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
        path = data_dir / f'rhythm_{day}.jsonl'
        with open(path, mode) as f:
            for obs in day_obs:
                f.write(json.dumps(obs) + '\n')
        files_written.append(str(path))

    return files_written


if __name__ == '__main__':
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description='Rhythm data collector')
    parser.add_argument('--dry-run', action='store_true', help='generate synthetic data')
    parser.add_argument('--days', type=int, default=7, help='days of synthetic data (default: 7)')
    parser.add_argument('--interval', type=int, default=5, help='seconds between collections (default: 5)')
    parser.add_argument('--duration', type=int, default=0, help='total seconds to collect (0 = single shot)')
    parser.add_argument('--data-dir', default=None, help='output directory')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(__file__).resolve().parent.parent.parent / 'data')

    if args.dry_run:
        print(f"generating {args.days} days of synthetic rhythm data...")
        obs = generate_synthetic(days=args.days)
        files = save_observations(obs, args.data_dir, overwrite=True)
        print(f"wrote {len(obs)} observations across {len(files)} files:")
        for f in files:
            print(f"  {f}")
    elif args.duration > 0:
        total = args.duration // args.interval
        print(f"collecting {total} observations over {args.duration}s (every {args.interval}s)")
        tracker = ActivityTracker()
        batch = []
        saved_total = 0
        start = _time.time()
        try:
            i = 0
            while _time.time() - start < args.duration:
                obs = tracker.sample()
                if obs:
                    batch.append(obs)
                    i += 1
                    if i % 100 == 0 or i == 1:
                        print(f"  {i}/{total} | idle={obs['I']:.0f}s activity={obs['A']:.1f}/min")
                    if len(batch) >= 50:
                        save_observations(batch, args.data_dir)
                        saved_total += len(batch)
                        batch = []
                _time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\nstopped early after {saved_total + len(batch)} observations")
        if batch:
            save_observations(batch, args.data_dir)
            saved_total += len(batch)
        print(f"saved {saved_total} observations to {args.data_dir}")
    else:
        tracker = ActivityTracker()
        obs = tracker.sample()
        if obs is None:
            print("rhythm collection failed", file=sys.stderr)
            sys.exit(1)
        files = save_observations([obs], args.data_dir)
        print(json.dumps(obs, indent=2))
        print(f"saved -> {files[0]}")
