"""Nerve atom trainer — trains on cross-atom anomaly scores + user feedback."""

import sys
import json
import glob
import random
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from kiri.core import Atom, StateLanguage
from kiri.atoms.nerve.config import NERVE_SCHEMA, ACTIONS, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, LANGUAGE_NAME


def make_nerve_language():
    """Create Nerve language with action tokens appended to the schema tokens."""
    lang = StateLanguage(LANGUAGE_NAME, NERVE_SCHEMA)
    # Add action tokens to vocabulary
    for action in ACTIONS:
        token = f'A:{action}'
        lang.stoi[token] = len(lang.tokens)
        lang.itos[len(lang.tokens)] = token
        lang.tokens.append(token)
    lang.vocab_size = len(lang.tokens)
    return lang


def load_data(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files matching: {pattern}", file=sys.stderr)
        sys.exit(1)

    raw = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

    # Prefer human-labeled data over model-labeled data.
    # For each timestamp, if a human entry exists, drop model entries.
    human_ts = {obs['ts'] for obs in raw if obs.get('source') == 'human'}
    observations = [
        obs for obs in raw
        if obs.get('source') != 'model' or obs['ts'] not in human_ts
    ]

    n_human = sum(1 for o in observations if o.get('source') == 'human')
    n_model = sum(1 for o in observations if o.get('source') == 'model')
    n_unlabeled = len(observations) - n_human - n_model
    print(f"loaded {len(observations)} observations from {len(files)} files "
          f"(human={n_human}, model={n_model}, unlabeled={n_unlabeled})")
    return observations


def build_sequences(observations, lang, seq_len):
    """Build sequences with action tokens appended to each observation."""
    tokens_flat = []
    for obs in observations:
        encoded = lang.encode_observation({k: obs[k] for k in lang.schema if k in obs})
        tokens_flat.extend(encoded[1:])  # skip BOS per observation
        # Append action token if present
        action = obs.get('action', 'ok')
        action_token = f'A:{action}'
        if action_token in lang.stoi:
            tokens_flat.append(lang.stoi[action_token])

    sequences = []
    full = [lang.BOS] + tokens_flat
    stride = len(lang.schema) + 1  # +1 for action token

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


def predict_action(atom, nerve_obs):
    """Given a Nerve observation, predict the best action."""
    lang = atom.lang
    tokens = lang.encode_observation({k: nerve_obs[k] for k in lang.schema if k in nerve_obs})

    preds = atom.predict_next(tokens)
    # Filter to action tokens only
    action_preds = [(name, prob) for name, prob in preds if name.startswith('A:')]
    action_preds.sort(key=lambda x: -x[1])

    if action_preds:
        best_action = action_preds[0][0].replace('A:', '')
        return best_action, action_preds
    return 'ok', action_preds


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
    print("\n--- nerve decision test ---")

    # Normal scores → should predict 'ok'
    normal = {'P': 1.5, 'R': 1.0, 'D': 0.8, 'H': 14, 'W': 2}
    action_n, preds_n = predict_action(atom, normal)
    print(f"\nnormal scores (P=1.5, R=1.0, D=0.8):")
    print(f"  predicted action: {action_n}")
    for name, prob in preds_n[:4]:
        print(f"    {name:>12s}  prob={prob:.3f}")

    # High scores → should predict 'alert'
    high = {'P': 9.0, 'R': 6.0, 'D': 1.0, 'H': 3, 'W': 6}
    action_h, preds_h = predict_action(atom, high)
    print(f"\nhigh scores (P=9.0, R=6.0, D=1.0, 3am Sunday):")
    print(f"  predicted action: {action_h}")
    for name, prob in preds_h[:4]:
        print(f"    {name:>12s}  prob={prob:.3f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the Nerve atom')
    parser.add_argument('--data', required=True, help='glob pattern for JSONL data files')
    parser.add_argument('--steps', type=int, default=500, help='training steps (default: 500)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--name', default='nerve', help='model name (default: nerve)')
    parser.add_argument('--resume', default=None, help='path to existing weights')
    parser.add_argument('--verbose', action='store_true', help='run decision test after training')
    args = parser.parse_args()

    lang = make_nerve_language()
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
    # Save language manually since it has extra action tokens
    import json as _json
    lang_data = {
        'name': lang.name, 'schema': lang.schema,
        'actions': ACTIONS, 'vocab_size': lang.vocab_size,
        'tokens': lang.tokens
    }
    with open(str(lang_path), 'w') as f:
        _json.dump(lang_data, f)

    print(f"\nsaved weights -> {weights_path}")
    print(f"saved language -> {lang_path}")

    if args.verbose:
        run_anomaly_comparison(atom, lang)
