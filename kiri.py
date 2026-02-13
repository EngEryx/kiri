"""KIRI daemon — collects from all atoms, scores, decides, acts."""

import sys
import math
import json
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR, COLLECT_INTERVAL, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from .core import Atom, StateLanguage, Pipe
from .atoms.pulse.collect import collect_local, save_observations as save_pulse
from .atoms.rhythm.collect import ActivityTracker, save_observations as save_rhythm
from .atoms.drift.collect import save_observations as save_drift
from .atoms.nerve.collect import score_atoms, log_feedback
from .atoms.nerve.train import predict_action, make_nerve_language
from .daemon.scheduler import Scheduler
from .daemon.alerts import send_telegram


def _load_atom(weights_path, lang_path):
    """Load a trained atom from disk. Returns (atom, lang) or (None, None)."""
    try:
        lang = StateLanguage.load(str(lang_path))
        atom = Atom(lang)
        atom.load_weights(str(weights_path))
        return atom, lang
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None


def _load_nerve(weights_path, lang_path):
    """Load Nerve atom with action tokens."""
    try:
        lang = make_nerve_language()
        atom = Atom(lang)
        atom.load_weights(str(weights_path))
        return atom, lang
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None


class KiriDaemon:
    """Main daemon: collect, score, decide, act."""

    def __init__(self, data_dir=None):
        self.data_dir = str(data_dir or DATA_DIR)
        self.rhythm_tracker = ActivityTracker()
        self.feedback_log = []

        # Load trained atoms (if weights exist)
        base = Path(__file__).resolve().parent
        self.pulse_atom, self.pulse_lang = _load_atom(
            base / 'atoms/pulse/weights/pulse_weights.json',
            base / 'atoms/pulse/weights/pulse_lang.json',
        )
        self.rhythm_atom, self.rhythm_lang = _load_atom(
            base / 'atoms/rhythm/weights/rhythm_weights.json',
            base / 'atoms/rhythm/weights/rhythm_lang.json',
        )
        self.drift_atom, self.drift_lang = _load_atom(
            base / 'atoms/drift/weights/drift_weights.json',
            base / 'atoms/drift/weights/drift_lang.json',
        )
        self.nerve_atom, self.nerve_lang = _load_nerve(
            base / 'atoms/nerve/weights/nerve_weights.json',
            base / 'atoms/nerve/weights/nerve_lang.json',
        )

        loaded = sum(1 for a in [self.pulse_atom, self.rhythm_atom, self.drift_atom, self.nerve_atom] if a)
        print(f"loaded {loaded}/4 trained atoms")

    def _score_sequence(self, atom, obs_dict):
        """Average anomaly score for an observation through an atom."""
        if atom is None:
            return 0.0
        tokens = atom.lang.encode_observation(obs_dict)
        if len(tokens) < 2:
            return 0.0
        n = min(atom.block_size, len(tokens) - 1)
        keys = [[] for _ in range(atom.n_layer)]
        vals = [[] for _ in range(atom.n_layer)]
        scores = []
        for pos in range(n):
            logits = atom.forward(tokens[pos], pos, keys, vals)
            probs = atom._softmax(logits)
            target = tokens[pos + 1]
            target_prob = probs[target].data
            scores.append(-math.log(max(target_prob, 1e-10)))
        return sum(scores) / len(scores) if scores else 0.0

    def collect_cycle(self):
        """One full collection → score → decide → act cycle."""
        now = datetime.now()

        # Collect pulse
        pulse_obs = collect_local()
        if pulse_obs:
            save_pulse([pulse_obs], self.data_dir)

        # Collect rhythm
        rhythm_obs = self.rhythm_tracker.sample()
        if rhythm_obs:
            save_rhythm([rhythm_obs], self.data_dir)

        # Score through atoms
        pulse_score = self._score_sequence(self.pulse_atom, pulse_obs) if pulse_obs else 0.0
        rhythm_score = self._score_sequence(self.rhythm_atom, rhythm_obs) if rhythm_obs else 0.0

        # Nerve decision
        action = 'ok'
        if self.nerve_atom:
            nerve_obs = {
                'P': pulse_score, 'R': rhythm_score, 'D': 0.0,
                'H': now.hour, 'W': now.weekday(),
                'ts': now.isoformat()
            }
            action, _ = predict_action(self.nerve_atom, nerve_obs)
            log_feedback(nerve_obs, action, self.data_dir)

        # Act
        status = f"P={pulse_score:.1f} R={rhythm_score:.1f} -> {action}"
        if action == 'alert' and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            msg = f"KIRI alert: {status}"
            if pulse_obs:
                msg += f"\nCPU={pulse_obs['C']:.0f}% MEM={pulse_obs['M']:.0f}%"
            send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
            print(f"  ALERT sent: {status}")
        elif action == 'alert':
            print(f"  ALERT (no telegram configured): {status}")
        elif action == 'suppress':
            print(f"  suppressed: {status}")
        else:
            print(f"  ok: {status}")

    def run(self):
        """Start the daemon."""
        print(f"kiri daemon starting")
        print(f"collecting every {COLLECT_INTERVAL} minutes -> {self.data_dir}")

        scheduler = Scheduler()
        scheduler.every(COLLECT_INTERVAL, self.collect_cycle, name='kiri-cycle')
        scheduler.run_forever()


def main():
    daemon = KiriDaemon()
    daemon.run()


if __name__ == '__main__':
    main()
