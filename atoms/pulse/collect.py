"""Pulse data collector — gathers infrastructure metrics from the local Mac Mini."""

import os
import sys
import json
import random
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path


def _run(cmd):
    """Run a command and return stdout, or None on failure."""
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def collect_local():
    """Collect live metrics from the local macOS system. Returns observation dict or None."""
    try:
        # CPU usage via top (1 sample, 0 processes)
        top_out = _run(["top", "-l", "1", "-n", "0", "-s", "0"])
        cpu = 0.0
        if top_out:
            m = re.search(r'CPU usage:\s+([\d.]+)% user,\s+([\d.]+)% sys', top_out)
            if m:
                cpu = float(m.group(1)) + float(m.group(2))

        # Memory via vm_stat
        vm_out = _run(["vm_stat"])
        mem = 0.0
        if vm_out:
            pages = {}
            for line in vm_out.splitlines():
                if ':' in line:
                    key, val = line.split(':', 1)
                    val = val.strip().rstrip('.')
                    if val.isdigit():
                        pages[key.strip()] = int(val)
            page_size = 16384  # macOS default page size
            ps_out = _run(["sysctl", "-n", "hw.pagesize"])
            if ps_out and ps_out.isdigit():
                page_size = int(ps_out)
            total_out = _run(["sysctl", "-n", "hw.memsize"])
            total_bytes = int(total_out) if total_out and total_out.isdigit() else 1
            active = pages.get('Pages active', 0)
            wired = pages.get('Pages wired down', 0)
            compressed = pages.get('Pages occupied by compressor', 0)
            used_bytes = (active + wired + compressed) * page_size
            mem = (used_bytes / total_bytes) * 100

        # Disk usage via df
        df_out = _run(["df", "-k", "/"])
        disk = 0.0
        if df_out:
            lines = df_out.splitlines()
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 5:
                    # capacity column (e.g. "45%")
                    cap = parts[4].rstrip('%')
                    if cap.isdigit():
                        disk = float(cap)

        # Swap via sysctl
        swap_out = _run(["sysctl", "vm.swapusage"])
        swap = 0.0
        if swap_out:
            used_m = re.search(r'used\s*=\s*([\d.]+)M', swap_out)
            total_m = re.search(r'total\s*=\s*([\d.]+)M', swap_out)
            if used_m and total_m:
                total_swap = float(total_m.group(1))
                if total_swap > 0:
                    swap = (float(used_m.group(1)) / total_swap) * 100

        # Load average
        load_out = _run(["sysctl", "-n", "vm.loadavg"])
        load_avg = 0.0
        if load_out:
            # format: "{ 1.23 4.56 7.89 }"
            nums = re.findall(r'[\d.]+', load_out)
            if nums:
                load_avg = float(nums[0])  # 1-minute load

        # Network check: can we reach a DNS root?
        net = 0
        try:
            import socket
            socket.create_connection(('1.1.1.1', 53), timeout=2).close()
            net = 1
        except OSError:
            pass

        now = datetime.now()
        return {
            'C': max(0, min(100, cpu)),
            'M': max(0, min(100, mem)),
            'D': max(0, min(100, disk)),
            'S': max(0, min(100, swap)),
            'L': max(0, min(20, load_avg)),
            'N': net,
            'H': now.hour,
            'ts': now.isoformat()
        }

    except Exception as e:
        print(f"local collection failed: {e}", file=sys.stderr)
        return None


def generate_synthetic(days=7, interval_minutes=5):
    """Generate synthetic infrastructure data for testing."""
    observations = []
    start = datetime.now() - timedelta(days=days)
    steps = (days * 24 * 60) // interval_minutes

    random.seed(42)
    disk_base = 35.0

    for i in range(steps):
        ts = start + timedelta(minutes=i * interval_minutes)
        hour = ts.hour
        weekday = ts.weekday() < 5

        is_work = weekday and 8 <= hour <= 18
        is_night = hour < 6 or hour >= 22

        if is_work:
            cpu = random.gauss(55, 12)
            mem = random.gauss(60, 10)
            load = random.gauss(4.0, 1.5)
            swap = random.gauss(15, 5)
        elif is_night:
            cpu = random.gauss(8, 4)
            mem = random.gauss(25, 5)
            load = random.gauss(0.5, 0.3)
            swap = random.gauss(5, 3)
        else:
            cpu = random.gauss(20, 8)
            mem = random.gauss(35, 8)
            load = random.gauss(1.5, 0.8)
            swap = random.gauss(8, 4)

        disk = disk_base + (i / max(steps, 1)) * 10 + random.gauss(0, 1)
        net = 0 if random.random() < 0.02 else 1

        obs = {
            'C': max(0, min(100, cpu)),
            'M': max(0, min(100, mem)),
            'D': max(0, min(100, disk)),
            'S': max(0, min(100, swap)),
            'L': max(0, min(20, load)),
            'N': net,
            'H': hour,
            'ts': ts.isoformat()
        }
        observations.append(obs)

    return observations


def save_observations(observations, data_dir, overwrite=False):
    """Save observations as JSONL, one file per day.
    Default appends to existing files (daemon-safe). Use overwrite=True for clean writes."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    by_day = {}
    for obs in observations:
        day = obs['ts'][:10]
        by_day.setdefault(day, []).append(obs)

    mode = 'w' if overwrite else 'a'
    files_written = []
    for day, day_obs in sorted(by_day.items()):
        path = data_dir / f'pulse_{day}.jsonl'
        with open(path, mode) as f:
            for obs in day_obs:
                f.write(json.dumps(obs) + '\n')
        files_written.append(str(path))

    return files_written


if __name__ == '__main__':
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description='Pulse data collector')
    parser.add_argument('--dry-run', action='store_true', help='generate synthetic data instead of live collection')
    parser.add_argument('--days', type=int, default=7, help='days of synthetic data (default: 7)')
    parser.add_argument('--interval', type=int, default=5, help='seconds between collections (default: 5)')
    parser.add_argument('--duration', type=int, default=0, help='total seconds to collect (0 = single shot)')
    parser.add_argument('--data-dir', default=None, help='output directory for JSONL files')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(__file__).resolve().parent.parent.parent / 'data')

    if args.dry_run:
        print(f"generating {args.days} days of synthetic pulse data...")
        obs = generate_synthetic(days=args.days)
        files = save_observations(obs, args.data_dir, overwrite=True)
        print(f"wrote {len(obs)} observations across {len(files)} files:")
        for f in files:
            print(f"  {f}")
    elif args.duration > 0:
        # Continuous collection mode
        total = args.duration // args.interval
        print(f"collecting {total} observations over {args.duration}s (every {args.interval}s)")
        batch = []
        saved_total = 0
        start = _time.time()
        try:
            i = 0
            while _time.time() - start < args.duration:
                obs = collect_local()
                if obs:
                    batch.append(obs)
                    i += 1
                    if i % 100 == 0 or i == 1:
                        print(f"  {i}/{total} | C={obs['C']:.0f}% M={obs['M']:.0f}% D={obs['D']:.0f}%")
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
        # Single shot
        obs = collect_local()
        if obs is None:
            print("local collection failed", file=sys.stderr)
            sys.exit(1)
        files = save_observations([obs], args.data_dir)
        print(json.dumps(obs, indent=2))
        print(f"saved -> {files[0]}")
