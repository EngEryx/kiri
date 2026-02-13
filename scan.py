"""
kiri scan — collect, train, and analyze your Mac in one shot.

Usage:
    python3 -m kiri.scan              # 60 samples, 1s apart
    python3 -m kiri.scan --samples 30 # fewer samples, faster
    python3 -m kiri.scan --train-only --steps 500  # retrain on existing data
"""
import sys, time, json, os, glob as _glob

# ensure parent is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiri.atoms.pulse.collect import collect_local as pulse_collect
from kiri.atoms.rhythm.collect import ActivityTracker
from kiri.atoms.pulse import config as pulse_cfg
from kiri.atoms.rhythm import config as rhythm_cfg
from kiri.core import Atom, StateLanguage


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def collect_batch(n, interval):
    """Collect n observations from Pulse and Rhythm."""
    pulse_obs, rhythm_obs = [], []
    tracker = ActivityTracker(idle_threshold=300)
    for i in range(n):
        p = pulse_collect()
        r = tracker.sample()
        if p:
            pulse_obs.append(p)
        if r:
            rhythm_obs.append(r)
        pct = int((i + 1) / n * 100)
        bar = '#' * (pct // 2) + '-' * (50 - pct // 2)
        sys.stdout.write(f"\r  collecting [{bar}] {i+1}/{n}")
        sys.stdout.flush()
        if i < n - 1:
            time.sleep(interval)
    print()
    return pulse_obs, rhythm_obs


def load_from_disk(prefix):
    """Load observations from existing JSONL files."""
    pattern = os.path.join(DATA_DIR, f'{prefix}_*.jsonl')
    files = sorted(_glob.glob(pattern))
    if not files:
        return []
    observations = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    observations.append(json.loads(line))
    print(f"  {prefix}: loaded {len(observations)} observations from {len(files)} files")
    return observations


def train_and_score(name, schema_def, n_embd, n_head, n_layer, block_size, observations, steps=0):
    """Train on observations and return anomaly scores."""
    lang = StateLanguage(name, schema_def)
    seq_len = block_size

    # encode all observations into token sequences
    tokens = []
    for obs in observations:
        tokens.extend(lang.encode_observation(obs))

    # build training sequences
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len // 2):
        sequences.append(tokens[i:i + seq_len])

    if not sequences:
        print(f"  {name}: not enough data ({len(tokens)} tokens, need {seq_len})")
        return None, None, None

    # train
    model = Atom(lang, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=seq_len)

    if steps == 0:
        steps = min(300, len(sequences) * 50)
    print(f"  {name}: {len(sequences)} sequences, {lang.vocab_size} vocab, training {steps} steps...")

    losses = []
    for step in range(steps):
        seq = sequences[step % len(sequences)]
        loss = model.train_step(seq, lr=0.01 * max(0.1, 1 - step / steps))
        losses.append(loss)
        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / 50
            sys.stdout.write(f"\r  step {step+1}/{steps}  loss {avg:.3f}")
            sys.stdout.flush()
    print()

    # score every observation window — anomaly_score returns (score, top3, actual)
    scores = []
    for seq in sequences:
        s, _, _ = model.anomaly_score(seq)
        scores.append(s)

    return model, lang, scores


def analyze(name, observations, scores, schema_def):
    """Print analysis of scores and current state."""
    if scores is None:
        return

    avg = sum(scores) / len(scores)
    lo = min(scores)
    hi = max(scores)

    # adaptive threshold: mean + 1.5 * std
    if len(scores) > 1:
        variance = sum((s - avg) ** 2 for s in scores) / (len(scores) - 1)
        std = variance ** 0.5
        threshold = avg + 1.5 * std
    else:
        threshold = hi

    print(f"  scores: min={lo:.2f}  avg={avg:.2f}  max={hi:.2f}  threshold={threshold:.2f}")

    # flag anomalous windows
    anomalies = [(i, s) for i, s in enumerate(scores) if s > threshold]
    if anomalies:
        print(f"  anomalous windows: {len(anomalies)}/{len(scores)}")
        for i, s in anomalies[:3]:
            print(f"    window {i}: score {s:.2f}")
    else:
        print(f"  no anomalies detected (all {len(scores)} windows normal)")

    # current values summary
    latest = observations[-1]
    print(f"\n  latest observation:")
    for k, v in latest.items():
        if k == 'ts':
            continue
        if isinstance(v, float):
            print(f"    {k}: {v:.1f}")
        else:
            print(f"    {k}: {v}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='KIRI scan — collect, train, analyze')
    parser.add_argument('--samples', type=int, default=60, help='observations to collect (default 60)')
    parser.add_argument('--interval', type=float, default=1.0, help='seconds between samples (default 1)')
    parser.add_argument('--train-only', action='store_true', help='skip collection, train on existing data')
    parser.add_argument('--steps', type=int, default=0, help='training steps (0 = auto)')
    args = parser.parse_args()

    banner("KIRI SCAN")

    if args.train_only:
        # load from disk
        print(f"  loading existing data from {DATA_DIR}\n")
        banner("PHASE 1: LOAD")
        pulse_obs = load_from_disk('pulse')
        rhythm_obs = load_from_disk('rhythm')
    else:
        print(f"  collecting {args.samples} samples, {args.interval}s apart")
        print(f"  estimated time: {int(args.samples * args.interval)}s\n")
        banner("PHASE 1: COLLECT")
        pulse_obs, rhythm_obs = collect_batch(args.samples, args.interval)
        print(f"\n  pulse: {len(pulse_obs)} observations")
        print(f"  rhythm: {len(rhythm_obs)} observations")

    # 2. train
    banner("PHASE 2: TRAIN")
    p_model, p_lang, p_scores = train_and_score(
        "pulse", pulse_cfg.PULSE_SCHEMA,
        pulse_cfg.N_EMBD, pulse_cfg.N_HEAD, pulse_cfg.N_LAYER, pulse_cfg.BLOCK_SIZE,
        pulse_obs, steps=args.steps)
    r_model, r_lang, r_scores = train_and_score(
        "rhythm", rhythm_cfg.RHYTHM_SCHEMA,
        rhythm_cfg.N_EMBD, rhythm_cfg.N_HEAD, rhythm_cfg.N_LAYER, rhythm_cfg.BLOCK_SIZE,
        rhythm_obs, steps=args.steps)

    # 3. analyze
    banner("PHASE 3: ANALYZE — PULSE (infrastructure)")
    analyze("pulse", pulse_obs, p_scores, pulse_cfg.PULSE_SCHEMA)

    banner("PHASE 3: ANALYZE — RHYTHM (work pattern)")
    analyze("rhythm", rhythm_obs, r_scores, rhythm_cfg.RHYTHM_SCHEMA)

    # 4. verdict
    banner("VERDICT")
    issues = []
    if p_scores:
        p_avg = sum(p_scores) / len(p_scores)
        if p_avg > 3.0:
            issues.append(f"pulse avg score {p_avg:.1f} (elevated)")
    if r_scores:
        r_avg = sum(r_scores) / len(r_scores)
        if r_avg > 3.0:
            issues.append(f"rhythm avg score {r_avg:.1f} (elevated)")

    latest_p = pulse_obs[-1] if pulse_obs else {}
    if latest_p.get('C', 0) > 80:
        issues.append(f"CPU at {latest_p['C']:.0f}%")
    if latest_p.get('M', 0) > 90:
        issues.append(f"memory at {latest_p['M']:.0f}%")
    if latest_p.get('S', 0) > 90:
        issues.append(f"swap at {latest_p['S']:.0f}%")
    if latest_p.get('D', 0) > 85:
        issues.append(f"disk at {latest_p['D']:.0f}%")

    if issues:
        print("  ATTENTION:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  all clear. your mac looks healthy.")

    print()


if __name__ == '__main__':
    main()
