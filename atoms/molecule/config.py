"""Molecule configuration — MoE transformer vocabulary, hyperparams, scenario definitions."""

# Domain-prefixed schemas (p. = pulse, r. = rhythm, d. = drift)
PULSE_TOKENS = {
    'p.C': (0, 100, 10),   # CPU %
    'p.M': (0, 100, 10),   # Memory %
    'p.D': (0, 100, 10),   # Disk %
    'p.S': (0, 100, 5),    # Swap %
    'p.L': (0, 20, 5),     # Load average
    'p.N': (0, 1, 2),      # Network up/down
}

RHYTHM_TOKENS = {
    'r.I': (0, 3600, 8),   # Idle duration (seconds)
    'r.A': (0, 60, 6),     # Activity density
    'r.H': (0, 24, 8),     # Hour of day
    'r.W': (0, 7, 7),      # Day of week
}

DRIFT_TOKENS = {
    'd.T': (0, 20, 5),     # Tasks added
    'd.C': (0, 20, 5),     # Tasks completed
    'd.S': (0, 10, 5),     # Project switches
    'd.H': (0, 24, 8),     # Hour of day
    'd.W': (0, 7, 3),      # Day of week
}

SCORE_TOKENS = {
    'PS': (0, 20, 5),      # Pulse anomaly score
    'RS': (0, 20, 5),      # Rhythm anomaly score
    'DS': (0, 20, 5),      # Drift anomaly score
}

TEMPORAL_TOKENS = {
    'H': (0, 24, 8),       # Hour of day
    'W': (0, 7, 5),        # Day of week
}

# Actions
ACTIONS = ['ok', 'alert', 'suppress', 'retrain']

# Control tokens
CONTROL_TOKENS = ['<SEP>', '<EXP>', '<END>']

# Explanation vocabulary (~50 words)
EXPLANATION_WORDS = [
    'high', 'low', 'normal', 'spike', 'drop', 'idle', 'active', 'busy',
    'cpu', 'memory', 'disk', 'swap', 'load', 'network',
    'night', 'morning', 'afternoon', 'evening', 'weekend', 'weekday',
    'task', 'switch', 'drift', 'rhythm', 'pulse', 'nerve',
    'unusual', 'expected', 'elevated', 'critical', 'stable',
    'at', 'during', 'while', 'and', 'but', 'no', 'many', 'few',
    'detected', 'pattern', 'anomaly', 'score', 'above', 'below',
    'working', 'resting', 'overloaded', 'recovering',
]

# Model hyperparams
N_EMBD = 48
N_HEAD = 4
N_LAYER = 3
BLOCK_SIZE = 32
N_EXPERTS = 4
TOP_K = 2
FFN_DIM = 96

LANGUAGE_NAME = 'molecule'

# Scenario definitions for synthetic data
SCENARIOS = {
    # --- ok: 70% total ---
    'normal': {
        'weight': 0.45, 'action': 'ok',
        'explanations': [
            ['normal', 'cpu', 'memory', 'stable'],
            ['expected', 'pattern', 'detected'],
            ['normal', 'load', 'stable'],
            ['expected', 'rhythm', 'pattern'],
            ['normal', 'pulse', 'stable'],
        ],
    },
    'normal_idle': {
        'weight': 0.15, 'action': 'ok',
        'explanations': [
            ['idle', 'resting', 'expected'],
            ['low', 'load', 'normal', 'stable'],
            ['normal', 'idle', 'expected', 'pattern'],
        ],
    },
    'normal_busy': {
        'weight': 0.10, 'action': 'ok',
        'explanations': [
            ['busy', 'but', 'expected', 'pattern'],
            ['active', 'working', 'normal'],
            ['elevated', 'load', 'expected', 'working'],
        ],
    },
    # --- alert: 15% total ---
    'cpu_spike': {
        'weight': 0.05, 'action': 'alert',
        'explanations': [
            ['spike', 'cpu', 'high', 'load', 'detected'],
            ['cpu', 'high', 'elevated', 'load'],
            ['spike', 'cpu', 'above', 'normal'],
            ['high', 'cpu', 'load', 'unusual'],
        ],
    },
    'memory_pressure': {
        'weight': 0.03, 'action': 'alert',
        'explanations': [
            ['memory', 'high', 'swap', 'elevated'],
            ['high', 'memory', 'swap', 'detected'],
            ['memory', 'elevated', 'above', 'normal'],
        ],
    },
    'night_activity': {
        'weight': 0.03, 'action': 'alert',
        'explanations': [
            ['unusual', 'active', 'during', 'night'],
            ['active', 'at', 'night', 'unusual'],
            ['night', 'activity', 'detected', 'unusual'],
        ],
    },
    'task_overload': {
        'weight': 0.02, 'action': 'alert',
        'explanations': [
            ['many', 'task', 'switch', 'few', 'completed'],
            ['task', 'drift', 'many', 'switch'],
            ['high', 'task', 'switch', 'detected'],
        ],
    },
    'network_issue': {
        'weight': 0.02, 'action': 'alert',
        'explanations': [
            ['network', 'drop', 'detected', 'unusual'],
            ['network', 'drop', 'anomaly', 'detected'],
            ['unusual', 'network', 'pattern'],
        ],
    },
    # --- suppress: 10% total ---
    'suppress_known': {
        'weight': 0.05, 'action': 'suppress',
        'explanations': [
            ['elevated', 'cpu', 'during', 'weekend', 'expected'],
            ['expected', 'pattern', 'during', 'weekend'],
            ['elevated', 'load', 'expected', 'pattern'],
        ],
    },
    'suppress_routine': {
        'weight': 0.05, 'action': 'suppress',
        'explanations': [
            ['elevated', 'but', 'expected', 'pattern'],
            ['score', 'above', 'normal', 'but', 'expected'],
            ['pattern', 'detected', 'expected', 'stable'],
        ],
    },
    # --- retrain: 5% total ---
    'retrain_signal': {
        'weight': 0.03, 'action': 'retrain',
        'explanations': [
            ['pattern', 'drift', 'unusual', 'score', 'above', 'normal'],
            ['unusual', 'pattern', 'drift', 'detected'],
            ['anomaly', 'score', 'elevated', 'drift'],
        ],
    },
    'retrain_novel': {
        'weight': 0.02, 'action': 'retrain',
        'explanations': [
            ['unusual', 'pattern', 'no', 'expected', 'score'],
            ['anomaly', 'pattern', 'unusual', 'detected'],
            ['score', 'elevated', 'unusual', 'pattern'],
        ],
    },
}
