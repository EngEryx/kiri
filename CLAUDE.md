# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is KIRI

KIRI (Intelligent Runtime Inspector) is a composable tiny intelligence system built on Karpathy's microgpt. It uses ~200-line transformer models ("atoms") trained on quantized state sequences to predict patterns and detect anomalies. Zero external dependencies — pure Python 3 with a custom autograd engine.

The core idea: instead of English tokens, train microgpt on **state tokens** (CPU buckets, memory buckets, load average, etc.). The model learns sequential patterns in your system's behavior and flags deviations.

## Architecture

The package lives in `kiri/` with the following structure:

```
kiri/
  core/                Core engine (extracted from root/atom.py)
    value.py           Scalar autograd engine (backpropagation)
    language.py        StateLanguage — quantizes values into token buckets
    atom.py            The transformer model (train, predict, anomaly score)
    pipe.py            Linear composition engine for chaining functions
  atoms/
    pulse/             Infrastructure metrics atom
      config.py        Schema (CPU, mem, disk, swap, load, network) and hyperparams
      collect.py       Mac Mini local stats (sysctl, vm_stat, df) + synthetic data gen
      train.py         CLI trainer with anomaly comparison
      weights/         Saved model weights (JSON)
  daemon/
    scheduler.py       Simple interval scheduler (time.sleep loop)
    alerts.py          Telegram notification delivery
  data/                Collected JSONL observation files
  config.py            Global config (env vars for Telegram, paths)
  kiri.py              Daemon entry point
```

Legacy files in `root/`:
- **`root/atom.py`** — Original monolithic implementation. Preserved as reference. The core modules in `kiri/core/` are extracted from this file.
- **`root/atom_demo.py`** — Independent standalone demo. Uses a functional style with a smaller model. Not derived from `atom.py` — it's its own thing.
- **`root/kiri.html`** — Design document / landing page.

## Running

```bash
# Run from parent directory of kiri/

# Core imports
python3 -c "from kiri.core import Value, Atom, StateLanguage, Pipe; print('OK')"

# Generate synthetic data (1 week, 5-min intervals)
python3 -m kiri.atoms.pulse.collect --dry-run

# Collect live from Mac Mini
python3 -m kiri.atoms.pulse.collect

# Train on collected data
python3 -m kiri.atoms.pulse.train --data 'kiri/data/pulse_*.jsonl' --steps 500 --verbose

# Run original monolithic demo
python3 root/atom.py
```

No pip install needed. No dependencies beyond Python 3 stdlib.

## Key Concepts

**State language schema** defines the vocabulary. Each prefix maps to `(min_val, max_val, num_buckets)`:
```python
{'C': (0, 100, 10), 'M': (0, 100, 10), 'N': (0, 1, 2)}
```

**Anomaly scoring** works by measuring surprise: how much probability did the model assign to the token that actually appeared? High negative log probability = anomalous.

**The Pipe** is the composition mechanism. Atoms are pipe stages. Data flows: observe -> predict -> decide -> act. The loop variant feeds output back to input for self-improvement.

## Planned Atom Types

4 core atoms sharing the same codebase, differentiated by their `StateLanguage` schema and data collectors:
- **Pulse** — infrastructure metrics (Mac Mini local stats, MikroTik RouterOS)
- **Rhythm** — developer behavior (git, tasks, focus)
- **Nerve** — cross-atom decision engine (trained on other atoms' outputs + user feedback)
- **Bodyguard** — network security (MikroTik firewall logs)

Additional atoms may follow (Cashflow for financial patterns, etc.).

## Development Notes

- `kiri/core/` is the canonical modular implementation; `root/atom.py` is the original monolith kept as reference
- `root/atom_demo.py` is independent — not a slim version of `atom.py`
- Model weights serialize to JSON (not pickle) — both weights and Adam optimizer state
- The `Value` autograd class uses `__slots__` for memory efficiency
- Training uses Adam with linear LR decay and a 10% floor
- RMSNorm is used instead of LayerNorm (no learnable params)
- All matrix operations are pure Python nested lists of `Value` objects — no numpy
