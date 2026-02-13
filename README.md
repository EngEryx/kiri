# KIRI

Intelligent Runtime Inspector. Composable tiny transformers trained on quantized state sequences to predict system patterns and detect anomalies.

Pure Python 3. Zero dependencies (PyTorch optional for MPS acceleration). Runs on a Mac Mini forever.

**[Read the full explainer](https://engeryx.github.io/kiri/)**

Built on Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) ([blog post](http://karpathy.github.io/2026/02/12/microgpt/)) — 202 lines, 161 lines of code, 4,192 default params. KIRI replaces English tokens with **state tokens**: CPU buckets, memory buckets, load averages, disk usage. The model learns what's normal and flags what isn't.

## Quick Start

```bash
# run from parent directory of kiri/

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

# 3. collect live from your Mac
python3 -m kiri.atoms.pulse.collect
python3 -m kiri.atoms.pulse.collect --interval 1 --duration 3600  # blast mode

# 4. log task activity
python3 -m kiri.atoms.drift.collect --added 3 --completed 1 --switched 2

# 5. run daemon (all atoms, scheduled collection, Nerve decisions)
python3 -m kiri.kiri
```

## The Four Atoms

| Atom | What it watches | Data source | Anomaly example |
|------|----------------|-------------|-----------------|
| **Pulse** | Infrastructure health | Mac Mini: `top`, `vm_stat`, `df`, `sysctl` + MikroTik REST API | CPU 95% at 3am (score 7.98 vs normal 0.72) |
| **Rhythm** | Work patterns | `ioreg` HID idle time (keyboard/mouse) | Active at 3am Sunday |
| **Drift** | Task patterns | Manual CLI logging | 8 tasks added, 0 completed, 5 switches (score 4.06 vs normal 0.47) |
| **Nerve** | Cross-atom decisions | Other atoms' anomaly scores | Predicts: ok, alert, suppress, retrain |

## Structure

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
      collect.py          Mac Mini local stats
      collect_mikrotik.py MikroTik RouterOS REST API
      config.py, train.py, weights/
    rhythm/               work patterns
      collect.py          keyboard/mouse idle time
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
  root/                   original monolithic implementation
```

## PyTorch Acceleration

With PyTorch + MPS (Apple Silicon):

```python
from kiri.core import AtomTorch  # drop-in replacement
# 500 steps: 17.6s (PyTorch/MPS) vs ~8min (pure Python) = 27x faster
```

Requires `torch` in a virtualenv. The pure Python `Atom` works everywhere with zero dependencies.

## How It Works

Each observation is quantized into discrete tokens via a **state language schema**:

```python
{'C': (0, 100, 10), 'M': (0, 100, 10), 'D': (0, 100, 10),
 'S': (0, 100, 5),  'L': (0, 20, 5),   'N': (0, 1, 2)}
```

CPU 52% becomes token `C5`. Memory 88% becomes `M8`. Sequences of these tokens are fed to a tiny transformer that learns temporal patterns.

Anomaly detection measures surprise: `-log(probability the model assigned to what actually happened)`. High score = the model has never seen this pattern before.

**The daemon loop**: collect from all sources -> encode -> score through each atom -> Nerve decides (ok/alert/suppress/retrain) -> act (Telegram alert or log). User feedback trains Nerve. The system improves by running.
