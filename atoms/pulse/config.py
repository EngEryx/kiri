"""Pulse atom configuration — infrastructure metrics schema and model hyperparams."""

# State language schema: 6 metrics, 42 tokens + BOS = 43 vocab
PULSE_SCHEMA = {
    'C': (0, 100, 10),   # CPU %: 10 buckets
    'M': (0, 100, 10),   # Memory %: 10 buckets
    'D': (0, 100, 10),   # Disk %: 10 buckets
    'S': (0, 100, 5),    # Swap %: 5 buckets
    'L': (0, 20, 5),     # Load average (1-min): 5 buckets
    'N': (0, 1, 2),      # Network: 0=down, 1=up
}

# Model hyperparams
N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
BLOCK_SIZE = 16

LANGUAGE_NAME = 'pulse'
