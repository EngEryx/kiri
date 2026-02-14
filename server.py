"""KIRI real-time monitoring server.

Usage:
    python3 -m kiri.server                          # start on port 7745
    python3 -m kiri.server --collect                 # + continuous collection
    python3 -m kiri.server --collect --interval 30   # collect every 30s
"""

import http.server
import json
import math
import os
import sys
import threading
import time
import glob as _glob
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

_PKG = Path(__file__).resolve().parent

from kiri.atoms.pulse.collect import collect_local as _pulse_collect
from kiri.atoms.pulse.collect import save_observations as _pulse_save
from kiri.atoms.rhythm.collect import ActivityTracker, save_observations as _rhythm_save
from kiri.atoms.nerve.collect import log_feedback as _nerve_log
from kiri.atoms.pulse import config as _pcfg
from kiri.atoms.rhythm import config as _rcfg
from kiri.core import Atom, StateLanguage

_DATA = _PKG / 'data'
_PW = _PKG / 'atoms' / 'pulse' / 'weights'
_RW = _PKG / 'atoms' / 'rhythm' / 'weights'


# ---------------------------------------------------------------------------
# Fast inference — plain floats, no Value/autograd overhead
# ---------------------------------------------------------------------------

def _fl(w, x):
    """Matrix-vector multiply: w @ x."""
    return [sum(a * b for a, b in zip(r, x)) for r in w]


def _fn(x):
    """RMSNorm."""
    ms = sum(v * v for v in x) / len(x)
    s = (ms + 1e-5) ** -0.5
    return [v * s for v in x]


def _fs(v):
    """Softmax over list of floats."""
    m = max(v)
    e = [math.exp(x - m) for x in v]
    t = sum(e)
    return [x / t for x in e]


def _ff(w, tok, pos, ks, vs, nl, nh, hd, bs):
    """Single-token forward pass with plain floats."""
    x = [a + b for a, b in zip(w['wte'][tok], w['wpe'][pos % bs])]
    x = _fn(x)
    for i in range(nl):
        xr = x
        x = _fn(x)
        q = _fl(w[f'l{i}.wq'], x)
        k = _fl(w[f'l{i}.wk'], x)
        v = _fl(w[f'l{i}.wv'], x)
        ks[i].append(k)
        vs[i].append(v)
        xa = []
        for h in range(nh):
            s = h * hd
            qh = q[s:s + hd]
            kh = [ki[s:s + hd] for ki in ks[i]]
            vh = [vi[s:s + hd] for vi in vs[i]]
            al = [sum(qh[j] * kh[t][j] for j in range(hd)) / hd ** 0.5
                  for t in range(len(kh))]
            aw = _fs(al)
            xa.extend(sum(aw[t] * vh[t][j] for t in range(len(vh)))
                      for j in range(hd))
        x = _fl(w[f'l{i}.wo'], xa)
        x = [a + b for a, b in zip(x, xr)]
        xr = x
        x = _fn(x)
        x = [max(0.0, v) for v in _fl(w[f'l{i}.f1'], x)]
        x = _fl(w[f'l{i}.f2'], x)
        x = [a + b for a, b in zip(x, xr)]
    return _fl(w['lm_head'], x)


def _score_seq(w, lang, tokens, nl, nh, hd, bs):
    """Score a token sequence. Returns (avg_score, {token_name: score})."""
    n = min(bs, len(tokens) - 1)
    if n < 1:
        return 0.0, {}
    ks = [[] for _ in range(nl)]
    vs = [[] for _ in range(nl)]
    pt = {}
    total = 0.0
    for p in range(n):
        logits = _ff(w, tokens[p], p, ks, vs, nl, nh, hd, bs)
        probs = _fs(logits)
        tgt = tokens[p + 1]
        s = -math.log(max(probs[tgt], 1e-10))
        nm = lang.decode_token(tgt)
        if nm != '<BOS>':
            pt[nm] = round(s, 3)
        total += s
    return round(total / n, 3), pt


def _extract(atom):
    """Extract float weights from Atom (strip autograd)."""
    return {k: [[v.data for v in row] for row in mat]
            for k, mat in atom.sd.items()}


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

def _clean(d):
    """Remove 'ts' from observation dict."""
    return {k: v for k, v in (d or {}).items() if k != 'ts'}


class KiriState:
    def __init__(self):
        self.start_time = time.time()
        self.tracker = ActivityTracker()
        self.lock = threading.Lock()

        self.pulse_lang = StateLanguage(_pcfg.LANGUAGE_NAME, _pcfg.PULSE_SCHEMA)
        self.rhythm_lang = StateLanguage(_rcfg.LANGUAGE_NAME, _rcfg.RHYTHM_SCHEMA)
        self.pulse_w = None
        self.rhythm_w = None
        self.pulse_params = 0
        self.rhythm_params = 0
        self.last_trained = None
        self.total_obs = 0
        self.total_anomalies = 0

        self._cache = None
        self._cache_t = 0
        self._stop = threading.Event()
        self._thread = None

        self._load_models()
        self._count_obs()

    def _load_models(self):
        for name, wdir, cfg in [('pulse', _PW, _pcfg), ('rhythm', _RW, _rcfg)]:
            wp = wdir / f'{name}_weights.json'
            lp = wdir / f'{name}_lang.json'
            if wp.exists() and lp.exists():
                lang = StateLanguage.load(str(lp))
                atom = Atom(lang, n_embd=cfg.N_EMBD, n_head=cfg.N_HEAD,
                            n_layer=cfg.N_LAYER, block_size=cfg.BLOCK_SIZE)
                atom.load_weights(str(wp))
                if name == 'pulse':
                    self.pulse_lang = lang
                    self.pulse_w = _extract(atom)
                    self.pulse_params = atom.num_params
                else:
                    self.rhythm_lang = lang
                    self.rhythm_w = _extract(atom)
                    self.rhythm_params = atom.num_params
                print(f"  {name}: loaded ({atom.num_params:,} params)")
            else:
                print(f"  {name}: no weights")

    def _count_obs(self):
        n = 0
        for f in _DATA.glob('*.jsonl'):
            with open(f) as fh:
                n += sum(1 for line in fh if line.strip())
        self.total_obs = n

    def _score_one(self, name, obs):
        """Score a single observation. Returns (score, per_token, verdict, token_str)."""
        lang = self.pulse_lang if name == 'pulse' else self.rhythm_lang
        w = self.pulse_w if name == 'pulse' else self.rhythm_w
        cfg = _pcfg if name == 'pulse' else _rcfg
        tokens = lang.encode_observation(obs)
        tok_str = lang.decode_sequence(tokens).replace('<BOS> ', '')
        if w is None:
            return 0.0, {}, 'not_trained', tok_str
        avg, pt = _score_seq(w, lang, tokens,
                             cfg.N_LAYER, cfg.N_HEAD,
                             cfg.N_EMBD // cfg.N_HEAD, cfg.BLOCK_SIZE)
        verdict = 'normal' if avg < 2.0 else ('elevated' if avg < 5.0 else 'anomaly')
        if verdict == 'anomaly':
            with self.lock:
                self.total_anomalies += 1
        return avg, pt, verdict, tok_str

    # --- API methods ---

    def status(self):
        now = time.time()
        if self._cache and now - self._cache_t < 1.5:
            return self._cache

        p_obs = _pulse_collect()
        with self.lock:
            r_obs = self.tracker.sample()

        if p_obs:
            ps, ppt, pv, ptk = self._score_one('pulse', p_obs)
        else:
            ps, ppt, pv, ptk = 0, {}, 'error', ''

        if r_obs:
            rs, rpt, rv, rtk = self._score_one('rhythm', r_obs)
        else:
            rs, rpt, rv, rtk = 0, {}, 'error', ''

        result = {
            'timestamp': datetime.now().isoformat(),
            'pulse': {
                'metrics': _clean(p_obs), 'tokens': ptk,
                'score': ps, 'per_token': ppt, 'verdict': pv,
            },
            'rhythm': {
                'metrics': _clean(r_obs), 'tokens': rtk,
                'score': rs, 'per_token': rpt, 'verdict': rv,
            },
            'stats': {
                'total_observations': self.total_obs,
                'total_anomalies': self.total_anomalies,
                'uptime_seconds': int(now - self.start_time),
                'last_trained': self.last_trained,
                'pulse_params': self.pulse_params,
                'rhythm_params': self.rhythm_params,
            },
        }
        self._cache = result
        self._cache_t = now
        return result

    def history(self, prefix, n=100):
        files = sorted(_glob.glob(str(_DATA / f'{prefix}_*.jsonl')))
        all_obs = []
        for f in files:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        all_obs.append(json.loads(line))
        obs_list = all_obs[-n:]

        lang = self.pulse_lang if prefix == 'pulse' else self.rhythm_lang
        w = self.pulse_w if prefix == 'pulse' else self.rhythm_w
        cfg = _pcfg if prefix == 'pulse' else _rcfg
        results = []
        for obs in obs_list:
            tokens = lang.encode_observation(obs)
            tok_str = lang.decode_sequence(tokens).replace('<BOS> ', '')
            if w is not None:
                avg, _ = _score_seq(w, lang, tokens,
                                    cfg.N_LAYER, cfg.N_HEAD,
                                    cfg.N_EMBD // cfg.N_HEAD, cfg.BLOCK_SIZE)
                verdict = 'normal' if avg < 2.0 else ('elevated' if avg < 5.0 else 'anomaly')
            else:
                avg, verdict = 0.0, 'not_trained'
            results.append({
                'timestamp': obs.get('ts', ''), 'tokens': tok_str,
                'score': avg, 'verdict': verdict, 'metrics': _clean(obs),
            })
        return results

    def collect_once(self):
        p_obs = _pulse_collect()
        with self.lock:
            r_obs = self.tracker.sample()
        if p_obs:
            _pulse_save([p_obs], str(_DATA))
            with self.lock:
                self.total_obs += 1
        if r_obs:
            _rhythm_save([r_obs], str(_DATA))
            with self.lock:
                self.total_obs += 1
        pr = rr = None
        if p_obs:
            s, pt, v, tk = self._score_one('pulse', p_obs)
            pr = {'metrics': _clean(p_obs), 'tokens': tk, 'score': s,
                  'verdict': v, 'timestamp': p_obs.get('ts', '')}
        if r_obs:
            s, pt, v, tk = self._score_one('rhythm', r_obs)
            rr = {'metrics': _clean(r_obs), 'tokens': tk, 'score': s,
                  'verdict': v, 'timestamp': r_obs.get('ts', '')}
        return {'pulse': pr, 'rhythm': rr}

    def train_atom(self, name, steps, cb):
        if name not in ('pulse', 'rhythm'):
            cb(error=f'unknown atom: {name}')
            return
        lang = self.pulse_lang if name == 'pulse' else self.rhythm_lang
        cfg = _pcfg if name == 'pulse' else _rcfg
        wdir = _PW if name == 'pulse' else _RW

        # load data
        files = sorted(_glob.glob(str(_DATA / f'{name}_*.jsonl')))
        observations = []
        for f in files:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        observations.append(json.loads(line))
        if not observations:
            cb(error=f'no data for {name}')
            return

        # build sequences
        flat = []
        for obs in observations:
            enc = lang.encode_observation({k: obs[k] for k in lang.schema if k in obs})
            flat.extend(enc[1:])
        full = [lang.BOS] + flat
        stride = len(lang.schema)
        seqs = []
        bs = cfg.BLOCK_SIZE
        for i in range(0, len(full) - bs, stride):
            s = full[i:i + bs + 1]
            if len(s) == bs + 1:
                seqs.append(s)
        if not seqs:
            cb(error=f'not enough data ({len(observations)} obs)')
            return

        import random
        random.shuffle(seqs)

        atom = Atom(lang, n_embd=cfg.N_EMBD, n_head=cfg.N_HEAD,
                    n_layer=cfg.N_LAYER, block_size=cfg.BLOCK_SIZE)
        wp = wdir / f'{name}_weights.json'
        if wp.exists():
            atom.load_weights(str(wp))

        cb(info=f'{len(seqs)} sequences, {atom.num_params} params, {len(observations)} obs')

        for step in range(steps):
            seq = seqs[step % len(seqs)]
            lr = 0.01 * max(0.1, 1 - step / max(steps, 1))
            loss = atom.train_step(seq, lr=lr)
            cb(step=step + 1, loss=loss, lr=lr, steps=steps)
            if step % 10 == 0:
                time.sleep(0.001)

        # save
        wdir.mkdir(parents=True, exist_ok=True)
        atom.save(str(wdir / f'{name}_weights.json'))
        lang.save(str(wdir / f'{name}_lang.json'))
        w = _extract(atom)
        with self.lock:
            if name == 'pulse':
                self.pulse_w = w
                self.pulse_params = atom.num_params
            else:
                self.rhythm_w = w
                self.rhythm_params = atom.num_params
            self.last_trained = datetime.now().isoformat()
            self._cache = None

        cb(done=True, final_loss=loss, total_steps=steps)

    def start_collection(self, interval):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def run():
            print(f"  collecting every {interval}s")
            while not self._stop.is_set():
                try:
                    self.collect_once()
                except Exception as e:
                    print(f"  collect error: {e}", file=sys.stderr)
                self._stop.wait(interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop_collection(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_STATE = None


class KiriHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self._cors()
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == '/api/status':
                self._json(_STATE.status())
            elif u.path == '/api/history':
                n = int(q.get('n', ['100'])[0])
                atom = q.get('atom', ['pulse'])[0]
                self._json(_STATE.history(atom, n))
            elif u.path == '/api/collect':
                self._json(_STATE.collect_once())
            else:
                self._json({'error': 'not found'}, 404)
        except Exception as e:
            self._json({'error': str(e)}, 500)

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length) if length else b''

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == '/api/feedback':
            try:
                body = self._read_body()
                data = json.loads(body) if body else {}
                action = data.get('action', 'ok')
                if action not in ('ok', 'alert', 'suppress', 'retrain'):
                    self._json({'error': f'invalid action: {action}'}, 400)
                    return
                ts = data.get('timestamp', datetime.now().isoformat())
                nerve_obs = {
                    'P': data.get('P', 0.0),
                    'R': data.get('R', 0.0),
                    'D': data.get('D', 0.0),
                    'H': data.get('H', datetime.now().hour),
                    'W': data.get('W', datetime.now().weekday()),
                    'ts': ts,
                }
                path = _nerve_log(nerve_obs, action, str(_DATA), source='human')
                self._json({'ok': True, 'action': action, 'source': 'human', 'file': path})
            except Exception as e:
                self._json({'error': str(e)}, 500)
            return
        elif u.path == '/api/train':
            name = q.get('atom', ['pulse'])[0]
            steps = int(q.get('steps', ['300'])[0])

            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self._cors()
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()

            def cb(**kw):
                try:
                    if 'error' in kw:
                        d = {'error': kw['error']}
                    elif kw.get('info'):
                        d = {'info': kw['info']}
                    elif kw.get('done'):
                        d = {'done': True,
                             'final_loss': round(kw['final_loss'], 4),
                             'steps': kw['total_steps']}
                    else:
                        d = {'step': kw['step'],
                             'loss': round(kw['loss'], 4),
                             'lr': round(kw['lr'], 6),
                             'total': kw['steps']}
                    self.wfile.write(json.dumps(d).encode() + b'\n')
                    self.wfile.flush()
                except Exception:
                    pass

            _STATE.train_atom(name, steps, cb)
        else:
            self._json({'error': 'not found'}, 404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _STATE
    import argparse

    parser = argparse.ArgumentParser(description='KIRI monitoring server')
    parser.add_argument('--port', type=int, default=7745)
    parser.add_argument('--collect', action='store_true',
                        help='enable background collection')
    parser.add_argument('--interval', type=int, default=60,
                        help='collection interval in seconds (default: 60)')
    args = parser.parse_args()

    print(f'\n  KIRI server · port {args.port}')
    print(f'  data: {_DATA}')
    _STATE = KiriState()

    if args.collect:
        _STATE.start_collection(args.interval)

    server = http.server.ThreadingHTTPServer(('', args.port), KiriHandler)
    print(f'\n  http://localhost:{args.port}/api/status')
    print(f'  dashboard: https://engeryx.github.io/kiri/monitor.html')
    print(f'  Ctrl+C to stop\n')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  stopping...')
        _STATE.stop_collection()
        server.shutdown()


if __name__ == '__main__':
    main()
