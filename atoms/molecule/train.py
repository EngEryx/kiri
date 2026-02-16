"""Molecule trainer — trains MoE transformer on synthetic observation->action->explanation data."""

import sys
import json
import glob
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kiri.core.molecule import MoleculeLanguage, Molecule
from kiri.atoms.molecule.config import (
    PULSE_TOKENS, RHYTHM_TOKENS, DRIFT_TOKENS, SCORE_TOKENS, TEMPORAL_TOKENS,
    ACTIONS, CONTROL_TOKENS, EXPLANATION_WORDS,
    N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, N_EXPERTS, TOP_K, FFN_DIM,
)


def make_molecule_language():
    """Create unified MoleculeLanguage from all domain schemas."""
    schema = {}
    schema.update(PULSE_TOKENS)
    schema.update(RHYTHM_TOKENS)
    schema.update(DRIFT_TOKENS)
    schema.update(SCORE_TOKENS)
    schema.update(TEMPORAL_TOKENS)
    return MoleculeLanguage(schema, ACTIONS, CONTROL_TOKENS, EXPLANATION_WORDS)


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

    print(f"loaded {len(raw)} observations from {len(files)} files")
    return raw


def build_sequences(observations, lang):
    """Build training sequences in the format:
    <BOS> [pulse] [rhythm] [drift] [scores] [temporal] A:action <EXP> word word word <END>
    """
    sequences = []
    for obs in observations:
        seq = [lang.BOS]

        # Encode each domain
        for domain_key, schema_group in [
            ('pulse', PULSE_TOKENS),
            ('rhythm', RHYTHM_TOKENS),
            ('drift', DRIFT_TOKENS),
            ('scores', SCORE_TOKENS),
            ('temporal', TEMPORAL_TOKENS),
        ]:
            domain_data = obs.get(domain_key, {})
            for prefix in schema_group:
                if prefix in domain_data:
                    seq.append(lang.encode_value(prefix, domain_data[prefix]))

        # Action token
        action = obs.get('action', 'ok')
        action_token = f'A:{action}'
        if action_token in lang.stoi:
            seq.append(lang.stoi[action_token])

        # <EXP> control token
        seq.append(lang.control_ids['<EXP>'])

        # Explanation words
        explanation = obs.get('explanation', [])
        for word in explanation:
            if word in lang.stoi:
                seq.append(lang.stoi[word])

        # <END> control token
        seq.append(lang.control_ids['<END>'])

        if len(seq) <= BLOCK_SIZE:
            sequences.append(seq)

    print(f"built {len(sequences)} training sequences (max length {BLOCK_SIZE})")
    return sequences


def train(sequences, lang, steps, lr, resume_path=None):
    model = Molecule(lang, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER,
                     block_size=BLOCK_SIZE, n_experts=N_EXPERTS, top_k=TOP_K,
                     ffn_dim=FFN_DIM)
    print(f"molecule: {model.num_params:,} params | vocab {lang.vocab_size} | device {model.device}")

    if resume_path:
        model.load_weights(resume_path)
        print(f"resumed from {resume_path}")

    batch_size = 32
    for step in range(steps):
        # Sample a batch
        batch = [sequences[random.randint(0, len(sequences) - 1)] for _ in range(batch_size)]

        # LR decay with 10% floor
        current_lr = lr * max(0.1, 1 - step / max(steps, 1))
        if hasattr(model, '_optimizer'):
            for pg in model._optimizer.param_groups:
                pg['lr'] = current_lr

        loss = model.train_step(batch, lr=current_lr)

        if (step + 1) % 50 == 0 or step == 0:
            print(f"  step {step+1:4d}/{steps} | loss {loss:.4f} | lr {current_lr:.5f}")

    return model


def run_explanation_test(model, lang):
    """Test explanation generation with sample observations."""
    print("\n--- molecule explanation test ---")

    # Normal observation
    normal_pulse = {'p.C': 20, 'p.M': 35, 'p.D': 45, 'p.S': 5, 'p.L': 1.5, 'p.N': 1}
    normal_rhythm = {'r.I': 300, 'r.A': 15, 'r.H': 14, 'r.W': 2}
    normal_drift = {'d.T': 3, 'd.C': 2, 'd.S': 1, 'd.H': 14, 'd.W': 2}
    normal_scores = {'PS': 1.2, 'RS': 1.0, 'DS': 0.8}
    normal_temporal = {'H': 14, 'W': 2}
    normal_obs = {**normal_pulse, **normal_rhythm, **normal_drift, **normal_scores, **normal_temporal}
    tokens = lang.encode_observation(normal_obs)
    action, probs = model.predict_action(tokens)
    explanation = model.explain(tokens)
    print(f"\nnormal: action={action}, explanation=\"{explanation}\"")

    # CPU spike
    spike_pulse = {'p.C': 92, 'p.M': 78, 'p.D': 50, 'p.S': 45, 'p.L': 15, 'p.N': 1}
    spike_rhythm = {'r.I': 100, 'r.A': 30, 'r.H': 10, 'r.W': 1}
    spike_drift = {'d.T': 2, 'd.C': 1, 'd.S': 0, 'd.H': 10, 'd.W': 1}
    spike_scores = {'PS': 8.5, 'RS': 1.5, 'DS': 0.9}
    spike_temporal = {'H': 10, 'W': 1}
    spike_obs = {**spike_pulse, **spike_rhythm, **spike_drift, **spike_scores, **spike_temporal}
    tokens = lang.encode_observation(spike_obs)
    action, probs = model.predict_action(tokens)
    explanation = model.explain(tokens)
    print(f"cpu spike: action={action}, explanation=\"{explanation}\"")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the Molecule')
    parser.add_argument('--data', required=True, help='glob pattern for JSONL data files')
    parser.add_argument('--steps', type=int, default=1000, help='training steps (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--resume', default=None, help='path to existing weights')
    parser.add_argument('--verbose', action='store_true', help='run explanation test after training')
    args = parser.parse_args()

    lang = make_molecule_language()
    observations = load_data(args.data)
    sequences = build_sequences(observations, lang)

    if not sequences:
        print("not enough data to build training sequences", file=sys.stderr)
        sys.exit(1)

    random.shuffle(sequences)
    model = train(sequences, lang, args.steps, args.lr, args.resume)

    weights_dir = Path(__file__).resolve().parent / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_path = weights_dir / 'molecule_weights.pt'
    lang_path = weights_dir / 'molecule_lang.json'

    model.save(str(weights_path))
    lang.save(str(lang_path))

    print(f"\nsaved weights -> {weights_path}")
    print(f"saved language -> {lang_path}")

    if args.verbose:
        run_explanation_test(model, lang)
