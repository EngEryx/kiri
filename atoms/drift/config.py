"""Drift atom configuration — task pattern schema and model hyperparams."""

# State language schema: 5 metrics, 26 tokens + BOS = 27 vocab
DRIFT_SCHEMA = {
    'T': (0, 20, 5),    # Tasks added: 5 buckets
    'C': (0, 20, 5),    # Tasks completed: 5 buckets
    'S': (0, 10, 5),    # Project switches: 5 buckets
    'H': (0, 24, 8),    # Hour of day: 8 buckets
    'W': (0, 7, 3),     # Day of week: 3 buckets (early-week, mid, late+weekend)
}

N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
BLOCK_SIZE = 16

LANGUAGE_NAME = 'drift'
