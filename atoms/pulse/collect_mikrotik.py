"""MikroTik RouterOS collector — gathers router metrics via REST API (RouterOS 7+)."""

import os
import sys
import json
import random
import ssl
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
import base64


def collect_mikrotik():
    """Collect live metrics from a MikroTik router. Returns observation dict or None."""
    host = os.environ.get('MIKROTIK_HOST')
    user = os.environ.get('MIKROTIK_USER')
    password = os.environ.get('MIKROTIK_PASS')

    if not all([host, user, password]):
        print("missing MIKROTIK_HOST, MIKROTIK_USER, or MIKROTIK_PASS env vars", file=sys.stderr)
        return None

    ctx = ssl.create_default_context()
    if os.environ.get('MIKROTIK_INSECURE', '').strip() == '1':
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    ca_path = os.environ.get('MIKROTIK_CA_CERT')
    if ca_path:
        ctx.load_verify_locations(ca_path)

    creds = base64.b64encode(f'{user}:{password}'.encode()).decode()
    headers = {'Authorization': f'Basic {creds}', 'Content-Type': 'application/json'}

    def api_get(path):
        req = Request(f'https://{host}/rest{path}', headers=headers)
        with urlopen(req, context=ctx, timeout=10) as resp:
            return json.loads(resp.read())

    try:
        # System resources
        res = api_get('/system/resource')
        cpu = float(res.get('cpu-load', 0))
        mem_free = int(res.get('free-memory', 0))
        mem_total = int(res.get('total-memory', 1))
        mem = ((mem_total - mem_free) / mem_total) * 100
        uptime = res.get('uptime', '0s')

        # Active connections
        conns = api_get('/ip/firewall/connection')
        conn_count = len(conns) if isinstance(conns, list) else 0

        # Interface traffic (ether1 as WAN)
        interfaces = api_get('/interface')
        bw_in, bw_out = 0, 0
        for iface in interfaces:
            if iface.get('name') == 'ether1':
                bw_in = int(iface.get('rx-byte', 0)) / (1024 * 1024)  # MB
                bw_out = int(iface.get('tx-byte', 0)) / (1024 * 1024)
                break

        now = datetime.now()
        return {
            'C': max(0, min(100, cpu)),
            'M': max(0, min(100, mem)),
            'I': max(0, min(10000, bw_in)),
            'O': max(0, min(10000, bw_out)),
            'K': max(0, min(5000, conn_count)),
            'N': 1,
            'H': now.hour,
            'ts': now.isoformat(),
            'src': 'mikrotik'
        }

    except (URLError, KeyError, ValueError, OSError) as e:
        print(f"mikrotik collection failed: {e}", file=sys.stderr)
        return None


def generate_synthetic(days=7, interval_minutes=5):
    """Generate synthetic MikroTik data for testing."""
    observations = []
    start = datetime.now() - timedelta(days=days)
    steps = (days * 24 * 60) // interval_minutes

    random.seed(99)

    for i in range(steps):
        ts = start + timedelta(minutes=i * interval_minutes)
        hour = ts.hour
        weekday = ts.weekday() < 5

        is_work = weekday and 8 <= hour <= 18
        is_night = hour < 6 or hour >= 22

        if is_work:
            cpu = random.gauss(35, 10)
            mem = random.gauss(55, 8)
            bw_in = random.gauss(500, 150)
            bw_out = random.gauss(200, 80)
            conns = random.randint(150, 400)
        elif is_night:
            cpu = random.gauss(5, 3)
            mem = random.gauss(30, 5)
            bw_in = random.gauss(20, 10)
            bw_out = random.gauss(10, 5)
            conns = random.randint(10, 50)
        else:
            cpu = random.gauss(15, 7)
            mem = random.gauss(40, 6)
            bw_in = random.gauss(100, 50)
            bw_out = random.gauss(50, 25)
            conns = random.randint(50, 150)

        net = 0 if random.random() < 0.01 else 1

        obs = {
            'C': max(0, min(100, cpu)),
            'M': max(0, min(100, mem)),
            'I': max(0, min(10000, bw_in)),
            'O': max(0, min(10000, bw_out)),
            'K': max(0, min(5000, conns)),
            'N': net,
            'H': hour,
            'ts': ts.isoformat(),
            'src': 'mikrotik'
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
        path = data_dir / f'mikrotik_{day}.jsonl'
        with open(path, mode) as f:
            for obs in day_obs:
                f.write(json.dumps(obs) + '\n')
        files_written.append(str(path))

    return files_written


if __name__ == '__main__':
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description='MikroTik data collector')
    parser.add_argument('--dry-run', action='store_true', help='generate synthetic data')
    parser.add_argument('--days', type=int, default=7, help='days of synthetic data (default: 7)')
    parser.add_argument('--interval', type=int, default=5, help='seconds between collections (default: 5)')
    parser.add_argument('--duration', type=int, default=0, help='total seconds to collect (0 = single shot)')
    parser.add_argument('--data-dir', default=None, help='output directory')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(__file__).resolve().parent.parent.parent / 'data')

    if args.dry_run:
        print(f"generating {args.days} days of synthetic mikrotik data...")
        obs = generate_synthetic(days=args.days)
        files = save_observations(obs, args.data_dir, overwrite=True)
        print(f"wrote {len(obs)} observations across {len(files)} files:")
        for f in files:
            print(f"  {f}")
    elif args.duration > 0:
        total = args.duration // args.interval
        print(f"collecting {total} observations over {args.duration}s (every {args.interval}s)")
        collected = []
        start = _time.time()
        try:
            i = 0
            while _time.time() - start < args.duration:
                obs = collect_mikrotik()
                if obs:
                    collected.append(obs)
                    i += 1
                    if i % 100 == 0 or i == 1:
                        print(f"  {i}/{total} | C={obs['C']:.0f}% M={obs['M']:.0f}%")
                _time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\nstopped early after {len(collected)} observations")
        if collected:
            files = save_observations(collected, args.data_dir)
            print(f"saved {len(collected)} observations across {len(files)} files")
    else:
        obs = collect_mikrotik()
        if obs is None:
            print("mikrotik collection failed", file=sys.stderr)
            sys.exit(1)
        files = save_observations([obs], args.data_dir)
        print(json.dumps(obs, indent=2))
        print(f"saved -> {files[0]}")
