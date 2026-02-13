"""Rhythm atom configuration — work pattern schema and model hyperparams."""

# State language schema: 4 metrics, 29 tokens + BOS = 30 vocab
RHYTHM_SCHEMA = {
    'I': (0, 3600, 8),   # Idle duration (seconds): 8 buckets (0-7.5min each)
    'A': (0, 60, 6),     # Activity density (events/min): 6 buckets
    'H': (0, 24, 8),     # Hour of day: 8 buckets (3h each)
    'W': (0, 7, 7),      # Day of week: 7 buckets (Mon=0..Sun=6)
}

N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
BLOCK_SIZE = 16

LANGUAGE_NAME = 'rhythm'
