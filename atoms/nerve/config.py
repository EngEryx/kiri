"""Nerve atom configuration — cross-atom decision engine schema and model hyperparams."""

# State language schema: 5 metrics from other atoms' scores, 25 tokens + BOS = 26 vocab
NERVE_SCHEMA = {
    'P': (0, 20, 5),    # Pulse anomaly score bucket
    'R': (0, 20, 5),    # Rhythm anomaly score bucket
    'D': (0, 20, 5),    # Drift anomaly score bucket
    'H': (0, 24, 5),    # Hour of day: 5 buckets
    'W': (0, 7, 5),     # Day of week: 5 buckets
}

# Action vocabulary (appended to state tokens)
ACTIONS = ['ok', 'alert', 'suppress', 'retrain']

N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
BLOCK_SIZE = 16

LANGUAGE_NAME = 'nerve'
