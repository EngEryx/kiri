"""Atom — a tiny GPT trained on state sequences."""

import math
import random

from .value import Value
from .language import StateLanguage


class Atom:
    """A tiny GPT trained on state sequences."""

    def __init__(self, lang, n_embd=32, n_head=4, n_layer=2, block_size=16):
        self.lang = lang
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.head_dim = n_embd // n_head
        self.vocab_size = lang.vocab_size

        # Initialize parameters
        mat = lambda nout, nin, std=0.08: [
            [Value(random.gauss(0, std)) for _ in range(nin)]
            for _ in range(nout)
        ]
        self.sd = {
            'wte': mat(self.vocab_size, n_embd),
            'wpe': mat(block_size, n_embd),
            'lm_head': mat(self.vocab_size, n_embd),
        }
        for i in range(n_layer):
            self.sd[f'l{i}.wq'] = mat(n_embd, n_embd)
            self.sd[f'l{i}.wk'] = mat(n_embd, n_embd)
            self.sd[f'l{i}.wv'] = mat(n_embd, n_embd)
            self.sd[f'l{i}.wo'] = mat(n_embd, n_embd)
            self.sd[f'l{i}.f1'] = mat(4 * n_embd, n_embd)
            self.sd[f'l{i}.f2'] = mat(n_embd, 4 * n_embd)

        self.params = [p for mat in self.sd.values() for row in mat for p in row]
        self.num_params = len(self.params)

        # Adam state
        self.m = [0.0] * self.num_params
        self.v = [0.0] * self.num_params
        self.step_count = 0

    def _linear(self, x, w):
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

    def _softmax(self, logits):
        max_val = max(v.data for v in logits)
        exps = [(v - max_val).exp() for v in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _rmsnorm(self, x):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]

    def forward(self, token_id, pos_id, keys, values):
        """Single token forward pass. Returns logits over vocab."""
        sd, hd, nh = self.sd, self.head_dim, self.n_head
        x = [t + p for t, p in zip(sd['wte'][token_id], sd['wpe'][pos_id % self.block_size])]
        x = self._rmsnorm(x)

        for li in range(self.n_layer):
            xr = x
            x = self._rmsnorm(x)
            q = self._linear(x, sd[f'l{li}.wq'])
            k = self._linear(x, sd[f'l{li}.wk'])
            v = self._linear(x, sd[f'l{li}.wv'])
            keys[li].append(k)
            values[li].append(v)
            xa = []
            for h in range(nh):
                hs = h * hd
                qh = q[hs:hs+hd]
                kh = [ki[hs:hs+hd] for ki in keys[li]]
                vh = [vi[hs:hs+hd] for vi in values[li]]
                al = [sum(qh[j]*kh[t][j] for j in range(hd)) / hd**0.5 for t in range(len(kh))]
                aw = self._softmax(al)
                ho = [sum(aw[t]*vh[t][j] for t in range(len(vh))) for j in range(hd)]
                xa.extend(ho)
            x = self._linear(xa, sd[f'l{li}.wo'])
            x = [a + b for a, b in zip(x, xr)]
            xr = x
            x = self._rmsnorm(x)
            x = self._linear(x, sd[f'l{li}.f1'])
            x = [xi.relu() for xi in x]
            x = self._linear(x, sd[f'l{li}.f2'])
            x = [a + b for a, b in zip(x, xr)]

        return self._linear(x, sd['lm_head'])

    def train_step(self, token_sequence, lr=0.01):
        """Train on one sequence. Returns loss."""
        n = min(self.block_size, len(token_sequence) - 1)
        keys = [[] for _ in range(self.n_layer)]
        vals = [[] for _ in range(self.n_layer)]
        losses = []

        for pos in range(n):
            logits = self.forward(token_sequence[pos], pos, keys, vals)
            probs = self._softmax(logits)
            target = token_sequence[pos + 1]
            losses.append(-probs[target].log())

        loss = (1 / n) * sum(losses)
        loss.backward()

        # Adam update
        self.step_count += 1
        lr_t = lr * (1 - self.step_count / 10000)  # decay over 10K steps
        lr_t = max(lr_t, lr * 0.1)  # floor at 10% of initial
        b1, b2, eps = 0.85, 0.99, 1e-8
        for i, p in enumerate(self.params):
            self.m[i] = b1 * self.m[i] + (1 - b1) * p.grad
            self.v[i] = b2 * self.v[i] + (1 - b2) * p.grad ** 2
            mh = self.m[i] / (1 - b1 ** self.step_count)
            vh = self.v[i] / (1 - b2 ** self.step_count)
            p.data -= lr_t * mh / (vh ** 0.5 + eps)
            p.grad = 0

        return loss.data

    def predict_next(self, token_sequence, temperature=0.5):
        """Given a sequence, predict probability distribution over next token."""
        n = min(self.block_size, len(token_sequence))
        keys = [[] for _ in range(self.n_layer)]
        vals = [[] for _ in range(self.n_layer)]

        for pos in range(n):
            logits = self.forward(token_sequence[pos], pos, keys, vals)

        probs = self._softmax([l / temperature for l in logits])
        return [(self.lang.decode_token(i), p.data) for i, p in enumerate(probs)]

    def anomaly_score(self, token_sequence):
        """How surprised is the model by the LAST token in the sequence?
        Returns (score, expected_top3, actual_token).
        High score = anomalous."""
        if len(token_sequence) < 2:
            return 0.0, [], ''

        context = token_sequence[:-1]
        actual = token_sequence[-1]
        preds = self.predict_next(context)

        # Sort by probability
        preds_sorted = sorted(preds, key=lambda x: -x[1])
        top3 = preds_sorted[:3]

        # Find probability assigned to actual token
        actual_name = self.lang.decode_token(actual)
        actual_prob = next((p for name, p in preds if name == actual_name), 0.001)

        # Anomaly score: negative log probability (surprise)
        score = -math.log(max(actual_prob, 1e-10))

        return score, top3, actual_name

    def save(self, path):
        """Save model weights to file."""
        import json
        data = {
            'config': {
                'n_embd': self.n_embd, 'n_head': self.n_head,
                'n_layer': self.n_layer, 'block_size': self.block_size,
            },
            'step_count': self.step_count,
            'weights': {k: [[v.data for v in row] for row in mat] for k, mat in self.sd.items()},
            'adam_m': self.m,
            'adam_v': self.v,
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_weights(self, path):
        """Load model weights from file."""
        import json
        with open(path) as f:
            data = json.load(f)
        for k, mat in data['weights'].items():
            for i, row in enumerate(mat):
                for j, val in enumerate(row):
                    self.sd[k][i][j].data = val
        self.m = data.get('adam_m', self.m)
        self.v = data.get('adam_v', self.v)
        self.step_count = data.get('step_count', 0)
