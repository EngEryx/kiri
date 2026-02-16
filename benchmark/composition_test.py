"""
KIRI Composition Benchmark — The Experiment

Proves that composing two independent micro-transformer atoms detects
anomalies that neither atom detects alone.

Six phases:
  1. Data generation   — 20,160 joint observations (7 days × 2/min × 1440 min/day)
  2. Anomaly injection — 150 anomalies in 3 classes (A: Pulse-only, B: Rhythm-only, C: cross-domain)
  3. Training          — 3 models: Pulse atom, Rhythm atom, Monolithic combined
  4. Scoring           — 6 methods: individual, max, max+divergence, sum, L2
  5. Analysis          — precision / recall / F1 per class and overall
  6. Report            — JSON results + matplotlib figures

Run:
    python3 -m kiri.benchmark.composition_test [--steps 1000] [--figures]
"""

import sys
import os
import json
import math
import random
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from kiri.core.language import StateLanguage
from kiri.core.atom_torch import AtomTorch
from kiri.atoms.pulse.config import PULSE_SCHEMA
from kiri.atoms.rhythm.config import RHYTHM_SCHEMA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / 'data'
WEIGHTS_DIR = BENCHMARK_DIR / 'weights'
RESULTS_DIR = BENCHMARK_DIR / 'results'

N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
BLOCK_SIZE = 16
BATCH_SIZE = 32

N_DAYS = 7
SAMPLES_PER_HOUR = 12  # every 5 min
TOTAL_OBS = N_DAYS * 24 * SAMPLES_PER_HOUR  # 20,160

ANOMALY_COUNT_A = 50   # Pulse-only
ANOMALY_COUNT_B = 50   # Rhythm-only
ANOMALY_COUNT_C = 50   # Cross-domain

ALPHA = 0.5  # divergence weight for max+alpha*|diff| method

SEED = 42

# Combined schema for monolithic model
COMBINED_SCHEMA = {}
COMBINED_SCHEMA.update(PULSE_SCHEMA)
COMBINED_SCHEMA.update(RHYTHM_SCHEMA)


# ---------------------------------------------------------------------------
# Phase 1 — Data Generation
# ---------------------------------------------------------------------------

def _diurnal_cpu(hour):
    """CPU follows a diurnal pattern: low at night, peak mid-day."""
    base = 15 + 35 * math.exp(-0.5 * ((hour - 14) / 4) ** 2)
    return base + random.gauss(0, 5)


def _diurnal_mem(hour):
    """Memory usage climbs during work hours."""
    base = 30 + 25 * math.exp(-0.5 * ((hour - 15) / 5) ** 2)
    return base + random.gauss(0, 3)


def _diurnal_load(hour):
    """Load average tracks CPU roughly."""
    base = 1.0 + 3.0 * math.exp(-0.5 * ((hour - 14) / 4) ** 2)
    return max(0, base + random.gauss(0, 0.5))


def _diurnal_idle(hour, weekday):
    """Idle time: low during work hours on weekdays, high at night/weekends."""
    if weekday >= 5:  # weekend
        return random.uniform(600, 3600)
    if 9 <= hour <= 18:
        return random.uniform(0, 300)
    return random.uniform(300, 3600)


def _diurnal_activity(hour, weekday):
    """Activity density: high during work hours on weekdays."""
    if weekday >= 5:
        return random.uniform(0, 5)
    if 9 <= hour <= 18:
        return random.uniform(15, 55)
    return random.uniform(0, 10)


def generate_observations():
    """Generate 20,160 normal joint observations over 7 days."""
    random.seed(SEED)
    observations = []

    for day in range(N_DAYS):
        weekday = day % 7
        for hour in range(24):
            for sample in range(SAMPLES_PER_HOUR):
                minute = sample * 5
                fractional_hour = hour + minute / 60.0

                obs = {
                    # Pulse fields
                    'C': max(0, min(100, _diurnal_cpu(fractional_hour))),
                    'M': max(0, min(100, _diurnal_mem(fractional_hour))),
                    'D': max(0, min(100, 35 + random.gauss(0, 2))),      # disk: stable
                    'S': max(0, min(100, 5 + random.gauss(0, 2))),       # swap: low
                    'L': max(0, min(20, _diurnal_load(fractional_hour))),
                    'N': 1,                                               # network: up
                    # Rhythm fields
                    'I': max(0, min(3600, _diurnal_idle(fractional_hour, weekday))),
                    'A': max(0, min(60, _diurnal_activity(fractional_hour, weekday))),
                    'H': fractional_hour,
                    'W': weekday,
                    # Metadata
                    '_day': day,
                    '_hour': hour,
                    '_minute': minute,
                    '_label': 'normal',
                    '_class': None,
                }
                observations.append(obs)

    print(f"phase 1: generated {len(observations)} normal observations ({N_DAYS} days)")
    return observations


# ---------------------------------------------------------------------------
# Phase 2 — Anomaly Injection
# ---------------------------------------------------------------------------

def inject_anomalies(observations):
    """Inject 150 anomalies into random positions. Returns labels array."""
    random.seed(SEED + 1)
    n = len(observations)
    indices = random.sample(range(n), ANOMALY_COUNT_A + ANOMALY_COUNT_B + ANOMALY_COUNT_C)
    random.shuffle(indices)

    a_idx = indices[:ANOMALY_COUNT_A]
    b_idx = indices[ANOMALY_COUNT_A:ANOMALY_COUNT_A + ANOMALY_COUNT_B]
    c_idx = indices[ANOMALY_COUNT_A + ANOMALY_COUNT_B:]

    # Class A — Pulse-only: CPU/memory/load spike, rhythm stays normal
    for i in a_idx:
        obs = observations[i]
        obs['C'] = random.uniform(85, 100)
        obs['M'] = random.uniform(80, 100)
        obs['L'] = random.uniform(12, 20)
        obs['S'] = random.uniform(40, 90)
        obs['_label'] = 'anomaly'
        obs['_class'] = 'A'

    # Class B — Rhythm-only: work activity at abnormal hours, infra normal
    for i in b_idx:
        obs = observations[i]
        obs['I'] = random.uniform(0, 60)          # very low idle (active)
        obs['A'] = random.uniform(30, 60)          # high activity
        obs['H'] = random.uniform(0, 5)            # midnight-5am
        obs['W'] = random.choice([5, 6])           # weekend
        obs['_label'] = 'anomaly'
        obs['_class'] = 'B'

    # Class C — Cross-domain: mildly anomalous in BOTH domains, neither enough alone.
    # Pattern: upper-range infra load + user idle during peak work hours.
    # Pulse alone: slightly elevated but within noise → moderate score, below threshold.
    # Rhythm alone: idle during work hours (could be break) → moderate score, below threshold.
    # Together: sum/L2 of two moderate scores crosses the composed threshold.
    for i in c_idx:
        obs = observations[i]
        # Pulse: upper end of normal (daytime mean CPU~45, mem~50)
        # Enough to make the model slightly surprised but not alarmed
        obs['C'] = random.uniform(55, 68)
        obs['M'] = random.uniform(58, 72)
        obs['L'] = random.uniform(4, 6.5)
        obs['S'] = random.uniform(12, 25)
        obs['D'] = random.uniform(33, 38)
        obs['N'] = 1
        # Rhythm: idle + inactive during work hours
        obs['I'] = random.uniform(900, 2400)        # idle 15-40 min
        obs['A'] = random.uniform(0, 5)              # very low activity
        obs['H'] = random.uniform(10, 16)            # mid-day
        obs['W'] = random.randint(0, 4)              # weekday
        obs['_label'] = 'anomaly'
        obs['_class'] = 'C'

    counts = {'A': len(a_idx), 'B': len(b_idx), 'C': len(c_idx)}
    print(f"phase 2: injected {sum(counts.values())} anomalies — A:{counts['A']} B:{counts['B']} C:{counts['C']}")
    return observations


# ---------------------------------------------------------------------------
# Phase 3 — Training
# ---------------------------------------------------------------------------

def build_sequences(observations, lang, field_keys, seq_len):
    """Build overlapping token sequences from observations."""
    tokens_flat = []
    for obs in observations:
        encoded = lang.encode_observation({k: obs[k] for k in field_keys if k in obs})
        tokens_flat.extend(encoded[1:])  # skip per-obs BOS

    full = [lang.BOS] + tokens_flat
    stride = len(field_keys)
    sequences = []

    for i in range(0, len(full) - seq_len, stride):
        seq = full[i:i + seq_len + 1]
        if len(seq) == seq_len + 1:
            sequences.append(seq)

    return sequences


def train_model(name, lang, sequences, steps, lr=0.01, verbose=True):
    """Train an AtomTorch model and return it."""
    atom = AtomTorch(lang, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE)
    if verbose:
        print(f"  {name}: {atom.num_params:,} params | vocab {lang.vocab_size} | {len(sequences)} seqs | device {atom.device}")

    random.seed(SEED + 2)
    random.shuffle(sequences)

    t0 = time.time()
    for step in range(steps):
        batch_start = (step * BATCH_SIZE) % len(sequences)
        batch = sequences[batch_start:batch_start + BATCH_SIZE]
        if len(batch) < 2:
            batch = sequences[:BATCH_SIZE]
        loss = atom.train_step(batch, lr=lr)

        if verbose and ((step + 1) % 100 == 0 or step == 0):
            elapsed = time.time() - t0
            print(f"    step {step+1:5d}/{steps} | loss {loss:.4f} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"    done in {elapsed:.1f}s | final loss {loss:.4f}")

    return atom


def train_all(observations, steps, verbose=True):
    """Train three models: Pulse atom, Rhythm atom, Monolithic."""
    print(f"phase 3: training 3 models ({steps} steps each)")

    # Filter to normal observations for training
    normal_obs = [o for o in observations if o['_label'] == 'normal']
    print(f"  training set: {len(normal_obs)} normal observations")

    # Pulse atom
    pulse_lang = StateLanguage('pulse', PULSE_SCHEMA)
    pulse_fields = list(PULSE_SCHEMA.keys())
    pulse_seqs = build_sequences(normal_obs, pulse_lang, pulse_fields, BLOCK_SIZE)
    pulse_atom = train_model('pulse', pulse_lang, pulse_seqs, steps, verbose=verbose)

    # Rhythm atom
    rhythm_lang = StateLanguage('rhythm', RHYTHM_SCHEMA)
    rhythm_fields = list(RHYTHM_SCHEMA.keys())
    rhythm_seqs = build_sequences(normal_obs, rhythm_lang, rhythm_fields, BLOCK_SIZE)
    rhythm_atom = train_model('rhythm', rhythm_lang, rhythm_seqs, steps, verbose=verbose)

    # Monolithic combined
    mono_lang = StateLanguage('monolithic', COMBINED_SCHEMA)
    mono_fields = list(COMBINED_SCHEMA.keys())
    mono_seqs = build_sequences(normal_obs, mono_lang, mono_fields, BLOCK_SIZE)
    mono_atom = train_model('monolithic', mono_lang, mono_seqs, steps, verbose=verbose)

    return pulse_atom, rhythm_atom, mono_atom, pulse_lang, rhythm_lang, mono_lang


# ---------------------------------------------------------------------------
# Phase 4 — Scoring
# ---------------------------------------------------------------------------

SCORE_WINDOW = 3  # preceding observations used as context


def _encode_window(window_obs, lang, field_keys):
    """Encode a window of observations into a single token sequence."""
    tokens = [lang.BOS]
    for obs in window_obs:
        encoded = lang.encode_observation({k: obs[k] for k in field_keys})
        tokens.extend(encoded[1:])  # skip per-obs BOS
    return tokens


def _score_target(atom, token_sequence, n_target_tokens):
    """Score only the LAST n_target_tokens in a windowed sequence.

    Uses the full sequence for context (preceding observations improve
    predictions), but only averages the surprise over the target
    observation's tokens. This avoids dilution from normal context.
    """
    _, per_token = atom.anomaly_score(token_sequence)
    if not per_token:
        return 0.0
    target = per_token[-n_target_tokens:]
    if not target:
        return 0.0
    return sum(s for _, s, _ in target) / len(target)


def score_all(observations, pulse_atom, rhythm_atom, mono_atom,
              pulse_lang, rhythm_lang, mono_lang, verbose=True):
    """Score all observations with windowed context.

    Each observation is scored using SCORE_WINDOW preceding observations
    as context, but only the TARGET observation's tokens contribute to
    the anomaly score. This matches training conditions (long sequences)
    while keeping scores focused on the observation of interest.
    """
    print(f"phase 4: scoring {len(observations)} observations (context={SCORE_WINDOW})")
    t0 = time.time()

    pulse_fields = list(PULSE_SCHEMA.keys())
    rhythm_fields = list(RHYTHM_SCHEMA.keys())
    mono_fields = list(COMBINED_SCHEMA.keys())
    n_pulse = len(pulse_fields)
    n_rhythm = len(rhythm_fields)
    n_mono = len(mono_fields)

    results = []
    for i, obs in enumerate(observations):
        start = max(0, i - SCORE_WINDOW + 1)
        window = observations[start:i + 1]

        pulse_tokens = _encode_window(window, pulse_lang, pulse_fields)
        rhythm_tokens = _encode_window(window, rhythm_lang, rhythm_fields)
        mono_tokens = _encode_window(window, mono_lang, mono_fields)

        s_pulse = _score_target(pulse_atom, pulse_tokens, n_pulse)
        s_rhythm = _score_target(rhythm_atom, rhythm_tokens, n_rhythm)
        s_mono = _score_target(mono_atom, mono_tokens, n_mono)

        s_max = max(s_pulse, s_rhythm)
        s_max_div = max(s_pulse, s_rhythm) + ALPHA * abs(s_pulse - s_rhythm)
        s_sum = s_pulse + s_rhythm
        s_l2 = math.sqrt(s_pulse ** 2 + s_rhythm ** 2)

        scores = {
            'pulse': s_pulse, 'rhythm': s_rhythm, 'monolithic': s_mono,
            'max': s_max, 'max_div': s_max_div, 'sum': s_sum, 'l2': s_l2,
        }
        results.append((obs, scores))

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  scored {i+1}/{len(observations)} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  scoring complete in {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Phase 5 — Analysis
# ---------------------------------------------------------------------------

def find_optimal_threshold(scored, method, target_class=None):
    """Find the threshold that maximizes F1 for a given method and optional class filter."""
    # Collect (score, is_anomaly) pairs
    pairs = []
    for obs, scores in scored:
        s = scores[method]
        if target_class:
            is_anom = obs['_class'] == target_class
        else:
            is_anom = obs['_label'] == 'anomaly'
        pairs.append((s, is_anom))

    pairs.sort(key=lambda x: x[0])
    scores_sorted = [p[0] for p in pairs]

    # Try thresholds at percentiles and between unique scores
    best_f1 = 0
    best_thresh = 0
    best_metrics = {}

    # Sample 200 candidate thresholds across the score range
    s_min = scores_sorted[0]
    s_max = scores_sorted[-1]
    if s_max <= s_min:
        return 0, {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0, 'tp': 0, 'fp': 0, 'fn': 0}

    candidates = [s_min + (s_max - s_min) * i / 200 for i in range(1, 200)]

    for thresh in candidates:
        tp = fp = fn = tn = 0
        for s, is_anom in pairs:
            predicted = s >= thresh
            if predicted and is_anom:
                tp += 1
            elif predicted and not is_anom:
                fp += 1
            elif not predicted and is_anom:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'threshold': round(thresh, 4),
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            }

    return best_thresh, best_metrics


def analyze(scored, verbose=True):
    """Compute precision/recall/F1 for each method, overall and per class."""
    print("phase 5: analyzing results")
    methods = ['pulse', 'rhythm', 'monolithic', 'max', 'max_div', 'sum', 'l2']
    classes = ['A', 'B', 'C']

    results = {}

    for method in methods:
        method_results = {}

        # Overall
        _, overall = find_optimal_threshold(scored, method)
        method_results['overall'] = overall

        # Per class — use the overall threshold to evaluate per-class performance
        thresh = overall['threshold']
        for cls in classes:
            tp = fp = fn = tn = 0
            for obs, scores in scored:
                s = scores[method]
                is_this_class = obs['_class'] == cls
                predicted = s >= thresh

                if predicted and is_this_class:
                    tp += 1
                elif predicted and not is_this_class and obs['_label'] == 'anomaly':
                    # predicted anomaly but wrong class — not counted for this class
                    pass
                elif predicted and obs['_label'] == 'normal':
                    fp += 1  # false positive (only count once across classes)
                elif not predicted and is_this_class:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            method_results[f'class_{cls}'] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'tp': tp, 'fn': fn,
            }

        results[method] = method_results

    if verbose:
        print(f"\n{'method':<14} {'overall F1':>10} {'class A':>10} {'class B':>10} {'class C':>10}")
        print('-' * 56)
        for method in methods:
            r = results[method]
            print(f"{method:<14} {r['overall']['f1']:>10.3f} "
                  f"{r['class_A']['f1']:>10.3f} "
                  f"{r['class_B']['f1']:>10.3f} "
                  f"{r['class_C']['f1']:>10.3f}")

    return results


# ---------------------------------------------------------------------------
# Phase 6 — Report
# ---------------------------------------------------------------------------

def compute_score_stats(scored, method):
    """Compute score statistics for normal vs anomalous observations."""
    normal_scores = [s[method] for obs, s in scored if obs['_label'] == 'normal']
    anomaly_scores = [s[method] for obs, s in scored if obs['_label'] == 'anomaly']

    def stats(arr):
        if not arr:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        arr_sorted = sorted(arr)
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / len(arr)
        return {
            'mean': round(mean, 4),
            'std': round(math.sqrt(var), 4),
            'min': round(arr_sorted[0], 4),
            'max': round(arr_sorted[-1], 4),
            'median': round(arr_sorted[len(arr) // 2], 4),
            'count': len(arr),
        }

    return {'normal': stats(normal_scores), 'anomaly': stats(anomaly_scores)}


def save_report(scored, analysis, elapsed_total, steps):
    """Save JSON report and optionally generate figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = ['pulse', 'rhythm', 'monolithic', 'max', 'max_div', 'sum', 'l2']

    report = {
        'experiment': 'KIRI Composition Benchmark',
        'config': {
            'total_observations': TOTAL_OBS,
            'anomalies': {'A': ANOMALY_COUNT_A, 'B': ANOMALY_COUNT_B, 'C': ANOMALY_COUNT_C},
            'training_steps': steps,
            'alpha': ALPHA,
            'n_embd': N_EMBD, 'n_head': N_HEAD, 'n_layer': N_LAYER,
            'block_size': BLOCK_SIZE, 'seed': SEED,
        },
        'score_statistics': {m: compute_score_stats(scored, m) for m in methods},
        'analysis': analysis,
        'elapsed_seconds': round(elapsed_total, 1),
    }

    report_path = RESULTS_DIR / 'composition_results.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nphase 6: saved report -> {report_path}")

    return report


def generate_figures(scored, analysis):
    """Generate matplotlib figures for the paper."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    methods = ['pulse', 'rhythm', 'monolithic', 'max', 'max_div', 'sum', 'l2']
    method_labels = {
        'pulse': 'Pulse only',
        'rhythm': 'Rhythm only',
        'monolithic': 'Monolithic',
        'max': 'max(P,R)',
        'max_div': 'max+α|Δ|',
        'sum': 'P+R',
        'l2': '√(P²+R²)',
    }

    # --- Figure 1: F1 bar chart by method and class ---
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = ['overall', 'class_A', 'class_B', 'class_C']
    class_labels = ['Overall', 'Class A\n(Pulse-only)', 'Class B\n(Rhythm-only)', 'Class C\n(Cross-domain)']
    x = range(len(methods))
    width = 0.18
    colors = ['#2d3436', '#e17055', '#0984e3', '#00b894']

    for ci, (cls, clabel) in enumerate(zip(classes, class_labels)):
        values = [analysis[m].get(cls, {}).get('f1', 0) for m in methods]
        offset = (ci - 1.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width, label=clabel, color=colors[ci])

    ax.set_ylabel('F1 Score')
    ax.set_title('Composition Benchmark: F1 by Method and Anomaly Class')
    ax.set_xticks(list(x))
    ax.set_xticklabels([method_labels[m] for m in methods], rotation=15)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'f1_by_method.png', dpi=150)
    plt.close(fig)
    print(f"  saved figure -> {RESULTS_DIR / 'f1_by_method.png'}")

    # --- Figure 2: Score distributions (normal vs anomaly) for key methods ---
    key_methods = ['pulse', 'rhythm', 'max_div', 'monolithic']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, method in zip(axes, key_methods):
        normal_scores = [s[method] for obs, s in scored if obs['_label'] == 'normal']
        anomaly_scores = [s[method] for obs, s in scored if obs['_label'] == 'anomaly']

        ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='#636e72', density=True)
        ax.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='#d63031', density=True)
        thresh = analysis[method]['overall']['threshold']
        ax.axvline(thresh, color='#fdcb6e', linestyle='--', linewidth=1.5, label=f'thresh={thresh:.2f}')
        ax.set_title(method_labels[method])
        ax.set_xlabel('Anomaly Score')
        ax.legend(fontsize=7)

    fig.suptitle('Score Distributions: Normal vs Anomaly', fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'score_distributions.png', dpi=150)
    plt.close(fig)
    print(f"  saved figure -> {RESULTS_DIR / 'score_distributions.png'}")

    # --- Figure 3: Class C detection — the key result ---
    fig, ax = plt.subplots(figsize=(8, 5))
    c_methods = methods
    c_f1 = [analysis[m].get('class_C', {}).get('f1', 0) for m in c_methods]
    c_recall = [analysis[m].get('class_C', {}).get('recall', 0) for m in c_methods]

    bar_x = range(len(c_methods))
    ax.bar([xi - 0.15 for xi in bar_x], c_f1, 0.3, label='F1', color='#0984e3')
    ax.bar([xi + 0.15 for xi in bar_x], c_recall, 0.3, label='Recall', color='#00b894')
    ax.set_xticks(list(bar_x))
    ax.set_xticklabels([method_labels[m] for m in c_methods], rotation=15)
    ax.set_ylabel('Score')
    ax.set_title('Cross-Domain Anomaly Detection (Class C)\nThe composition advantage')
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'class_c_detection.png', dpi=150)
    plt.close(fig)
    print(f"  saved figure -> {RESULTS_DIR / 'class_c_detection.png'}")

    # --- Figure 4: Per-class recall heatmap ---
    fig, ax = plt.subplots(figsize=(8, 4))
    recall_matrix = []
    for cls in ['class_A', 'class_B', 'class_C']:
        row = [analysis[m].get(cls, {}).get('recall', 0) for m in methods]
        recall_matrix.append(row)

    im = ax.imshow(recall_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_labels[m] for m in methods], rotation=15)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Class A (Pulse)', 'Class B (Rhythm)', 'Class C (Cross)'])
    for i in range(3):
        for j in range(len(methods)):
            ax.text(j, i, f'{recall_matrix[i][j]:.2f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, label='Recall')
    ax.set_title('Per-Class Recall by Method')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'recall_heatmap.png', dpi=150)
    plt.close(fig)
    print(f"  saved figure -> {RESULTS_DIR / 'recall_heatmap.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_weights(pulse_atom, rhythm_atom, mono_atom):
    """Save trained model weights."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    pulse_atom.save(str(WEIGHTS_DIR / 'pulse.pt'))
    rhythm_atom.save(str(WEIGHTS_DIR / 'rhythm.pt'))
    mono_atom.save(str(WEIGHTS_DIR / 'mono.pt'))
    print(f"  saved weights -> {WEIGHTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description='KIRI Composition Benchmark')
    parser.add_argument('--steps', type=int, default=1000, help='training steps per model (default: 1000)')
    parser.add_argument('--figures', action='store_true', help='generate matplotlib figures')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--quiet', action='store_true', help='minimal output')
    args = parser.parse_args()

    verbose = not args.quiet
    t_start = time.time()

    print("=" * 60)
    print("KIRI Composition Benchmark")
    print("=" * 60)

    # Phase 1
    observations = generate_observations()

    # Phase 2
    observations = inject_anomalies(observations)

    # Phase 3
    pulse_atom, rhythm_atom, mono_atom, pulse_lang, rhythm_lang, mono_lang = \
        train_all(observations, args.steps, verbose=verbose)
    save_weights(pulse_atom, rhythm_atom, mono_atom)

    # Phase 4
    scored = score_all(observations, pulse_atom, rhythm_atom, mono_atom,
                       pulse_lang, rhythm_lang, mono_lang, verbose=verbose)

    # Phase 5
    analysis = analyze(scored, verbose=verbose)

    # Phase 6
    elapsed = time.time() - t_start
    report = save_report(scored, analysis, elapsed, args.steps)

    if args.figures:
        generate_figures(scored, analysis)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_composed = max(
        ['max', 'max_div', 'sum', 'l2'],
        key=lambda m: analysis[m]['class_C']['f1']
    )
    best_individual = max(
        ['pulse', 'rhythm'],
        key=lambda m: analysis[m]['class_C']['f1']
    )

    print(f"  class C (cross-domain) F1:")
    print(f"    pulse alone:     {analysis['pulse']['class_C']['f1']:.3f}")
    print(f"    rhythm alone:    {analysis['rhythm']['class_C']['f1']:.3f}")
    print(f"    monolithic:      {analysis['monolithic']['class_C']['f1']:.3f}")
    print(f"    best composed:   {analysis[best_composed]['class_C']['f1']:.3f} ({best_composed})")
    print(f"  composition advantage: {analysis[best_composed]['class_C']['f1'] - analysis[best_individual]['class_C']['f1']:+.3f} F1 over best individual")
    print(f"  total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
