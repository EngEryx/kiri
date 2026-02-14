"""KIRI unified runner — single entry point.

Usage:
    python3 -m kiri.run                        # default: port 7745, collect every 30s
    python3 -m kiri.run --port 8080            # custom port
    python3 -m kiri.run --interval 10          # collect every 10s
    python3 -m kiri.run --retrain-every 50     # retrain molecule every 50 collections
"""

import sys
import threading
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def _bootstrap_molecule():
    """If molecule weights don't exist, generate synthetic data + train."""
    weights_dir = _PKG / 'atoms' / 'molecule' / 'weights'
    wp = weights_dir / 'molecule_weights.pt'
    lp = weights_dir / 'molecule_lang.json'

    if wp.exists() and lp.exists():
        return True

    try:
        import torch
    except ImportError:
        print("  molecule: torch not installed, skipping bootstrap")
        return False

    print("  molecule: no weights found, bootstrapping...")

    from kiri.atoms.molecule.collect import generate_synthetic, save_observations
    from kiri.atoms.molecule.train import (
        make_molecule_language, build_sequences, train,
    )

    data_dir = _PKG / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    print("  generating 2000 synthetic samples...")
    obs = generate_synthetic(n_samples=2000)
    save_observations(obs, str(data_dir), overwrite=True)

    lang = make_molecule_language()
    sequences = build_sequences(obs, lang)
    if not sequences:
        print("  molecule bootstrap: not enough sequences")
        return False

    import random
    random.shuffle(sequences)

    print("  training molecule (1000 steps)...")
    model = train(sequences, lang, steps=1000, lr=0.01)

    weights_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(wp))
    lang.save(str(lp))
    print(f"  molecule: {model.num_params:,} params on {model.device}")
    return True


def _retrain_loop(state, threshold, stop_event):
    """Background thread: retrain molecule when enough observations accumulate."""
    while not stop_event.is_set():
        stop_event.wait(10)
        if stop_event.is_set():
            break
        with state.lock:
            count = state.molecule_obs_since_retrain
        if count >= threshold:
            print(f"\n  auto-retrain: {count} observations accumulated, retraining molecule...")
            state.molecule_retrain(steps=500)


def main():
    import argparse
    import http.server

    parser = argparse.ArgumentParser(description='KIRI — unified runner')
    parser.add_argument('--port', type=int, default=7745)
    parser.add_argument('--interval', type=int, default=30,
                        help='collection interval in seconds (default: 30)')
    parser.add_argument('--retrain-every', type=int, default=50,
                        help='retrain molecule every N collections (default: 50)')
    args = parser.parse_args()

    print('\n  KIRI')
    print('  ────')

    # Bootstrap molecule if needed
    _bootstrap_molecule()

    # Import server components after bootstrap (so molecule weights exist)
    from kiri.server import KiriState, KiriHandler
    import kiri.server as _srv

    print(f'\n  starting server · port {args.port}')
    print(f'  data: {_PKG / "data"}')
    _srv._STATE = KiriState()

    # Start background collection
    _srv._STATE.start_collection(args.interval)
    print(f'  collecting every {args.interval}s')

    # Start auto-retrain thread
    stop = threading.Event()
    retrain_thread = threading.Thread(
        target=_retrain_loop, args=(_srv._STATE, args.retrain_every, stop),
        daemon=True,
    )
    retrain_thread.start()
    print(f'  auto-retrain: every {args.retrain_every} collections')

    server = http.server.ThreadingHTTPServer(('', args.port), KiriHandler)
    print(f'\n  dashboard: http://localhost:{args.port}/')
    print(f'  api:       http://localhost:{args.port}/api/status')
    print(f'  Ctrl+C to stop\n')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  stopping...')
        stop.set()
        _srv._STATE.stop_collection()
        server.shutdown()


if __name__ == '__main__':
    main()
