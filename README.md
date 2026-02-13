# KIRI

Intelligent Runtime Inspector. Composable tiny transformers trained on quantized state sequences to predict system patterns and detect anomalies.

Pure Python 3. Zero dependencies. 27,840 parameters. Runs on a Mac Mini forever.

Built on [Karpathy's microgpt.py](https://github.com/karpathy/microgpt) (Feb 2026) — 199 lines, 160 lines of code, 4,224 default params. KIRI replaces English tokens with **state tokens**: CPU buckets, memory buckets, load averages, disk usage. The model learns what's normal and flags what isn't.

## Quick Start

```bash
# run from parent directory of kiri/

# 1. generate 1 week of synthetic data (2,016 observations at 5-min intervals)
python3 -m kiri.atoms.pulse.collect --dry-run

# 2. train on synthetic data with anomaly comparison
python3 -m kiri.atoms.pulse.train --data 'kiri/data/pulse_*.jsonl' --steps 500 --verbose

# 3. collect live from your Mac (single observation)
python3 -m kiri.atoms.pulse.collect

# 4. blast data for initial training (1 obs/sec for 1 hour)
python3 -m kiri.atoms.pulse.collect --interval 1 --duration 3600

# 5. steady-state monitoring (every 5 min, daemon mode)
python3 -m kiri.kiri
```

## What Happens

Training loss drops from ~3.98 to ~0.59 in 500 steps. The `--verbose` flag runs an anomaly comparison after training:

```
normal (moderate load):   avg score 1.33
anomalous (maxed out):    avg score 6.15
```

Per-token breakdown shows exactly which metrics surprise the model — C9 (CPU 90-100%) and M9 (memory 90-100%) score 9-12, while D4 (disk 40%) scores 0.13 because it's common in training data.

## Structure

```
kiri/
  core/                 autograd + transformer + state language + pipe
    value.py            scalar autograd engine (backpropagation)
    language.py         quantizes continuous values into token buckets
    atom.py             decoder-only transformer with Adam (betas 0.85/0.99)
    pipe.py             linear composition: observe -> predict -> decide -> act
  atoms/pulse/          infrastructure metrics atom
    config.py           schema: CPU, mem, disk, swap, load, network (42 tokens)
    collect.py          macOS local stats via top, vm_stat, df, sysctl
    train.py            CLI trainer with per-token anomaly scoring
    weights/            saved model weights (JSON, not pickle)
  daemon/
    scheduler.py        interval scheduler (time.sleep loop)
    alerts.py           telegram notifications
  data/                 collected JSONL observation files
  config.py             global config (env vars)
  kiri.py               daemon entry point
  root/                 original monolithic implementation (reference)
```

## Data Collection

Pulse collects from the local macOS system using `os` and `subprocess`:

| Metric | Source | Buckets |
|--------|--------|---------|
| CPU % | `top -l 1` | 10 |
| Memory % | `vm_stat` + `sysctl hw.memsize` | 10 |
| Disk % | `df -k /` | 10 |
| Swap % | `sysctl vm.swapusage` | 5 |
| Load avg | `sysctl -n vm.loadavg` | 5 |
| Network | up/down | 2 |

No external APIs. No credentials. Works immediately on any Mac.

## How It Works

Each observation is quantized into discrete tokens via a **state language schema**:

```python
{'C': (0, 100, 10), 'M': (0, 100, 10), 'D': (0, 100, 10),
 'S': (0, 100, 5),  'L': (0, 20, 5),   'N': (0, 1, 2)}
```

CPU 52% becomes token `C5`. Memory 88% becomes `M8`. A sequence of these tokens is fed to a tiny transformer (2 layers, 4 heads, 32-dim embeddings) that learns temporal patterns.

Anomaly detection measures surprise: `-log(probability the model assigned to what actually happened)`. High score = the model has never seen this pattern before.

## Vision

KIRI atoms are composable. Planned atom types:

- **Pulse** — infrastructure metrics (Mac Mini local stats, MikroTik RouterOS)
- **Rhythm** — developer behavior (git commits, task completion, focus blocks)
- **Nerve** — cross-atom decision engine (trained on other atoms' outputs)
- **Bodyguard** — network security (MikroTik firewall logs)

Atoms chain via **Pipes**. Molecules = multiple atoms composed. Organisms = self-retraining loops where output feeds back to input.
