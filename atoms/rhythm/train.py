"""Rhythm atom trainer — trains on work pattern data."""

import sys
import json
import glob
import random
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from kiri.core import Atom, StateLanguage
from kiri.atoms.rhythm.config import RHYTHM_SCHEMA, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, LANGUAGE_NAME


def load_data(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files matching: {pattern}", file=sys.stderr)
        sys.exit(1)

    observations = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    observations.append(json.loads(line))

    print(f"loaded {len(observations)} observations from {len(files)} files")
    return observations


def build_sequences(observations, lang, seq_len):
    tokens_flat = []
    for obs in observations:
        encoded = lang.encode_observation({k: obs[k] for k in lang.schema if k in obs})
        tokens_flat.extend(encoded[1:])

    sequences = []
    full = [lang.BOS] + tokens_flat
    stride = len(lang.schema)

    for i in range(0, len(full) - seq_len, stride):
        seq = full[i:i + seq_len + 1]
        if len(seq) == seq_len + 1:
            sequences.append(seq)

    print(f"built {len(sequences)} training sequences (length {seq_len})")
    return sequences


def sequence_anomaly_score(atom, token_sequence):
    n = min(atom.block_size, len(token_sequence) - 1)
    if n < 1:
        return 0.0, []

    keys = [[] for _ in range(atom.n_layer)]
    vals = [[] for _ in range(atom.n_layer)]
    per_token = []

    for pos in range(n):
        logits = atom.forward(token_sequence[pos], pos, keys, vals)
        probs = atom._softmax(logits)
        target = token_sequence[pos + 1]
        target_prob = probs[target].data
        score = -math.log(max(target_prob, 1e-10))
        per_token.append((atom.lang.decode_token(target), score, target_prob))

    avg = sum(s for _, s, _ in per_token) / len(per_token)
    return avg, per_token


def train(sequences, lang, steps, lr, resume_path=None):
    atom = Atom(lang, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE)
    print(f"atom: {atom.num_params:,} params | vocab {lang.vocab_size}")

    if resume_path:
        atom.load_weights(resume_path)
        print(f"resumed from {resume_path} (step {atom.step_count})")

    for step in range(steps):
        seq = sequences[step % len(sequences)]
        loss = atom.train_step(seq, lr=lr)

        if (step + 1) % 50 == 0 or step == 0:
            print(f"  step {step+1:4d}/{steps} | loss {loss:.4f}")

    return atom


def run_anomaly_comparison(atom, lang):
    print("\n--- anomaly comparison ---")

    # Normal: afternoon work, moderate idle, Tuesday
    normal = lang.encode_observation({
        'I': 120, 'A': 5.0, 'H': 14, 'W': 1,
    })
    score_n, details_n = sequence_anomaly_score(atom, normal)

    # Anomalous: active at 3am on Sunday, no idle
    anomalous = lang.encode_observation({
        'I': 10, 'A': 12.0, 'H': 3, 'W': 6,
    })
    score_a, details_a = sequence_anomaly_score(atom, anomalous)

    print(f"\nnormal (afternoon work, Tuesday):")
    print(f"  state: {lang.decode_sequence(normal)}")
    print(f"  avg score: {score_n:.3f}")
    for tok, s, p in details_n:
        print(f"    {tok:>4s}  score={s:.3f}  prob={p:.3f}")

    print(f"\nanomalous (3am Sunday, high activity):")
    print(f"  state: {lang.decode_sequence(anomalous)}")
    print(f"  avg score: {score_a:.3f}")
    for tok, s, p in details_a:
        print(f"    {tok:>4s}  score={s:.3f}  prob={p:.3f}")

    if score_a > score_n:
        print(f"\nmodel correctly finds anomaly more surprising ({score_a:.3f} > {score_n:.3f})")
    else:
        print(f"\nmodel needs more training ({score_a:.3f} <= {score_n:.3f}), try --steps 2000")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the Rhythm atom')
    parser.add_argument('--data', required=True, help='glob pattern for JSONL data files')
    parser.add_argument('--steps', type=int, default=500, help='training steps (default: 500)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--name', default='rhythm', help='model name (default: rhythm)')
    parser.add_argument('--resume', default=None, help='path to existing weights')
    parser.add_argument('--verbose', action='store_true', help='run anomaly comparison after training')
    args = parser.parse_args()

    lang = StateLanguage(LANGUAGE_NAME, RHYTHM_SCHEMA)
    observations = load_data(args.data)
    sequences = build_sequences(observations, lang, BLOCK_SIZE)

    if not sequences:
        print("not enough data to build training sequences", file=sys.stderr)
        sys.exit(1)

    random.shuffle(sequences)
    atom = train(sequences, lang, args.steps, args.lr, args.resume)

    weights_dir = Path(__file__).resolve().parent / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_path = weights_dir / f'{args.name}_weights.json'
    lang_path = weights_dir / f'{args.name}_lang.json'

    atom.save(str(weights_path))
    lang.save(str(lang_path))
    print(f"\nsaved weights -> {weights_path}")
    print(f"saved language -> {lang_path}")

    if args.verbose:
        run_anomaly_comparison(atom, lang)
