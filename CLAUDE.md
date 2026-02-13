# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is KIRI

KIRI (Intelligent Runtime Inspector) is a composable tiny intelligence system built on Karpathy's microgpt. It uses ~200-line transformer models ("atoms") trained on quantized state sequences to predict patterns and detect anomalies. Zero external dependencies — pure Python 3 with a custom autograd engine.

The core idea: instead of English tokens, train microgpt on **state tokens** (CPU buckets, memory buckets, load average, etc.). The model learns sequential patterns in your system's behavior and flags deviations.

## Architecture

The package lives in `kiri/` with the following structure:

```
kiri/
  core/                   engine
    value.py              scalar autograd (backpropagation)
    language.py           quantizes values into token buckets
    atom.py               decoder-only transformer + Adam
    atom_torch.py         PyTorch port (MPS accelerated, 27x faster)
    pipe.py               linear composition engine
  atoms/
    pulse/                infrastructure metrics
      collect.py          Mac Mini local stats (top, vm_stat, df, sysctl)
      collect_mikrotik.py MikroTik RouterOS REST API
      config.py, train.py, weights/
    rhythm/               work patterns
      collect.py          keyboard/mouse idle time (ioreg HIDIdleTime)
      config.py, train.py, weights/
    drift/                task patterns
      collect.py          manual CLI task logging
      config.py, train.py, weights/
    nerve/                decision engine
      collect.py          aggregates atom scores
      train.py            trains on scores + user feedback
      config.py, weights/
  daemon/
    scheduler.py          interval scheduler
    alerts.py             telegram notifications
  data/                   collected JSONL files
  config.py               global config (env vars)
  kiri.py                 daemon: collect all -> score -> Nerve decides -> act
```

Legacy files in `root/`:
- **`root/atom.py`** — Original monolithic implementation. Preserved as reference. The core modules in `kiri/core/` are extracted from this file.
- **`root/atom_demo.py`** — Independent standalone demo. Uses a functional style with a smaller model. Not derived from `atom.py` — it's its own thing.
- **`root/kiri.html`** — Design document / landing page.

## Running

```bash
# Run from parent directory of kiri/

# 1. generate synthetic data for all atoms
python3 -m kiri.atoms.pulse.collect --dry-run
python3 -m kiri.atoms.rhythm.collect --dry-run
python3 -m kiri.atoms.drift.collect --dry-run
python3 -m kiri.atoms.nerve.collect --dry-run

# 2. train all atoms
python3 -m kiri.atoms.pulse.train --data 'kiri/data/pulse_*.jsonl' --steps 500 --verbose
python3 -m kiri.atoms.rhythm.train --data 'kiri/data/rhythm_*.jsonl' --steps 500 --verbose
python3 -m kiri.atoms.drift.train --data 'kiri/data/drift_*.jsonl' --steps 500 --verbose
python3 -m kiri.atoms.nerve.train --data 'kiri/data/nerve_*.jsonl' --steps 500 --verbose

# 3. collect live from Mac
python3 -m kiri.atoms.pulse.collect
python3 -m kiri.atoms.pulse.collect --interval 1 --duration 3600  # blast mode

# 4. log task activity
python3 -m kiri.atoms.drift.collect --added 3 --completed 1 --switched 2

# 5. run daemon (all atoms, scheduled collection, Nerve decisions)
python3 -m kiri.kiri
```

No pip install needed. No dependencies beyond Python 3 stdlib. PyTorch optional for MPS acceleration.

## The Four Atoms

| Atom | What it watches | Data source | Anomaly example |
|------|----------------|-------------|-----------------|
| **Pulse** | Infrastructure health | Mac Mini: `top`, `vm_stat`, `df`, `sysctl` + MikroTik REST API | CPU 95% at 3am (score 7.98 vs normal 0.72) |
| **Rhythm** | Work patterns | `ioreg` HID idle time (keyboard/mouse) | Active at 3am Sunday |
| **Drift** | Task patterns | Manual CLI logging | 8 tasks added, 0 completed, 5 switches (score 4.06 vs normal 0.47) |
| **Nerve** | Cross-atom decisions | Other atoms' anomaly scores | Predicts: ok, alert, suppress, retrain |

## Key Concepts

**State language schema** defines the vocabulary. Each prefix maps to `(min_val, max_val, num_buckets)`:
```python
{'C': (0, 100, 10), 'M': (0, 100, 10), 'D': (0, 100, 10),
 'S': (0, 100, 5),  'L': (0, 20, 5),   'N': (0, 1, 2)}
```

**Anomaly scoring** measures surprise: `-log(probability the model assigned to what actually happened)`. High score = the model has never seen this pattern before. Averaged across all tokens in a sequence.

**The Pipe** is the composition mechanism. Data flows: observe -> predict -> decide -> act. The loop variant feeds output back for self-improvement.

**The daemon loop**: collect from all sources -> encode -> score through each atom -> Nerve decides (ok/alert/suppress/retrain) -> act (Telegram alert or log). User feedback trains Nerve. The system improves by running.

## PyTorch Acceleration

With PyTorch + MPS (Apple Silicon):
```python
from kiri.core import AtomTorch  # drop-in replacement
# 500 steps: 17.6s (PyTorch/MPS) vs ~8min (pure Python) = 27x faster
```

Requires `torch` in a virtualenv. The pure Python `Atom` works everywhere with zero dependencies.

## Development Notes

- `kiri/core/` is the canonical modular implementation; `root/atom.py` is the original monolith kept as reference
- `root/atom_demo.py` is independent — not a slim version of `atom.py`
- Model weights serialize to JSON (not pickle) — both weights and Adam optimizer state
- The `Value` autograd class uses `__slots__` for memory efficiency
- Training uses Adam with linear LR decay and a 10% floor
- RMSNorm is used instead of LayerNorm (no learnable params)
- All matrix operations are pure Python nested lists of `Value` objects — no numpy
- AtomTorch uses nn.Module with batched training (batch_size=32) for MPS acceleration
