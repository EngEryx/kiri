"""
ATOM — The fundamental unit of KIRI.

A microgpt that doesn't speak English. It speaks STATE.
It learns sequences of encoded observations and predicts what comes next.
When prediction ≠ reality, that's a signal.
When prediction = reality, it understands your system.

The atom does ONE thing: given a history of state tokens, predict the next one.
Everything else — anomaly detection, automation, self-improvement — 
emerges from COMPOSING atoms and LOOPING output back to input.

Based on Karpathy's microgpt.py (Feb 2026).
Modified by Eryx Labs for structured state prediction.

Usage:
  # 1. Collect state data (see collectors.py)
  # 2. Train:  python atom.py train --data states.txt --name pulse
  # 3. Predict: python atom.py predict --name pulse --input "C45M32D88N1"
  # 4. Anomaly: python atom.py watch --name pulse --feed live
"""

import os, math, random, json, time, sys

# ============================================================
# THE AUTOGRAD ENGINE (from Karpathy, untouched)
# This IS backpropagation. Every scalar tracks its gradient.
# ============================================================

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children: build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# ============================================================
# THE STATE LANGUAGE
# ============================================================
# Instead of English characters, our vocabulary is STATE TOKENS.
# Each token encodes one observation about the world.
#
# Examples of "sentences" in state language:
#
#   Infrastructure: "C45 M32 D88 N1 T09"
#     C=CPU%, M=Mem%, D=Disk%, N=NetOK, T=Hour
#
#   Behavior: "G3 T2 F0 S1 H14"
#     G=GitCommits, T=TasksDone, F=FocusBlocks, S=ScopeNew, H=Hour
#
#   Action: "!C>90 A:alert R:ack"
#     !=anomaly, C>90=what, A:alert=action taken, R:ack=result
#
# The model doesn't know what these MEAN. It just learns:
# "after C45 M32, the next token is usually D85-D92"
# If reality says D30, that's anomalous.
#
# KEY INSIGHT: We quantize continuous values into discrete buckets.
# CPU 0-100% becomes tokens: C0 C1 C2 ... C9 (10% buckets)
# This keeps vocabulary small = model stays tiny.
# ============================================================

class StateLanguage:
    """Defines the vocabulary for a state language."""

    def __init__(self, name, schema):
        """
        schema: dict mapping prefix to (min_val, max_val, num_buckets)
        Example: {'C': (0, 100, 10), 'M': (0, 100, 10), 'N': (0, 1, 2)}

        This creates tokens: C0 C1 C2...C9 M0 M1...M9 N0 N1
        Plus: BOS (beginning of sequence)
        """
        self.name = name
        self.schema = schema
        self.tokens = ['<BOS>']
        self.prefix_map = {}  # prefix -> list of token indices

        for prefix, (lo, hi, buckets) in schema.items():
            bucket_tokens = []
            for b in range(buckets):
                token = f"{prefix}{b}"
                bucket_tokens.append(len(self.tokens))
                self.tokens.append(token)
            self.prefix_map[prefix] = bucket_tokens

        self.vocab_size = len(self.tokens)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for i, t in enumerate(self.tokens)}
        self.BOS = 0

    def encode_value(self, prefix, value):
        """Quantize a continuous value into a token id."""
        lo, hi, buckets = self.schema[prefix]
        bucket = int((value - lo) / (hi - lo) * buckets)
        bucket = max(0, min(buckets - 1, bucket))
        token = f"{prefix}{bucket}"
        return self.stoi[token]

    def decode_token(self, token_id):
        """Get the string representation of a token."""
        return self.itos.get(token_id, '?')

    def encode_observation(self, obs_dict):
        """Encode a dict of {prefix: value} into a token sequence."""
        tokens = [self.BOS]
        for prefix in self.schema:  # deterministic order
            if prefix in obs_dict:
                tokens.append(self.encode_value(prefix, obs_dict[prefix]))
        return tokens

    def decode_sequence(self, token_ids):
        """Decode token ids back to readable form."""
        return ' '.join(self.decode_token(t) for t in token_ids)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'name': self.name, 'schema': self.schema}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d['name'], {k: tuple(v) for k, v in d['schema'].items()})


# ============================================================
# THE MODEL (microgpt, configured for state prediction)
# ============================================================

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
        with open(path) as f:
            data = json.load(f)
        for k, mat in data['weights'].items():
            for i, row in enumerate(mat):
                for j, val in enumerate(row):
                    self.sd[k][i][j].data = val
        self.m = data.get('adam_m', self.m)
        self.v = data.get('adam_v', self.v)
        self.step_count = data.get('step_count', 0)


# ============================================================
# THE PIPE — How atoms compose
# ============================================================

class Pipe:
    """
    Connects atoms. Output of one atom's prediction feeds into
    another atom's input. This is how molecules form.

    Example pipe:
      observe -> predict -> compare -> act -> log -> observe (loop)

    Each stage is a function that takes input and returns output.
    An atom's predict_next IS a pipe stage.
    """

    def __init__(self):
        self.stages = []
        self.log = []

    def add(self, name, fn):
        """Add a stage. fn: takes input, returns output."""
        self.stages.append((name, fn))
        return self

    def run(self, initial_input):
        """Run the pipeline. Each stage's output feeds the next."""
        data = initial_input
        for name, fn in self.stages:
            data = fn(data)
            self.log.append({
                'stage': name,
                'output': str(data)[:200],
                'time': time.time()
            })
        return data

    def loop(self, initial_input, iterations=1, delay=0):
        """Run the pipeline in a loop, feeding output back to input."""
        data = initial_input
        for i in range(iterations):
            data = self.run(data)
            if delay > 0:
                time.sleep(delay)
        return data


# ============================================================
# DEMO: Train an atom on synthetic infrastructure data
# ============================================================

if __name__ == '__main__':

    print("=" * 60)
    print("ATOM — Tiny State Predictor")
    print("=" * 60)

    # Define a state language for infrastructure monitoring
    infra_lang = StateLanguage('infra', {
        'C': (0, 100, 10),   # CPU: 10 buckets (0-9%, 10-19%, ...)
        'M': (0, 100, 10),   # Memory: 10 buckets
        'D': (0, 100, 10),   # Disk: 10 buckets
        'N': (0, 1, 2),      # Network: 0=down, 1=up
        'H': (0, 24, 8),     # Hour of day: 3-hour buckets
    })

    print(f"\nLanguage: {infra_lang.name}")
    print(f"Vocabulary: {infra_lang.vocab_size} tokens")
    print(f"Tokens: {infra_lang.tokens}")

    # Generate synthetic training data (your Mac Mini would use real data)
    print("\n--- Generating synthetic state sequences ---")
    random.seed(42)
    docs = []

    for day in range(200):  # 200 "days" of data
        day_states = []
        for hour in range(0, 24, 3):  # 8 observations per day
            # Normal pattern: CPU/Mem correlate with work hours
            is_work = 9 <= hour <= 18
            cpu = random.gauss(55 if is_work else 15, 10)
            mem = random.gauss(45 if is_work else 25, 8)
            disk = 40 + day * 0.05 + random.gauss(0, 2)  # slowly growing
            net = 1 if random.random() > 0.02 else 0  # 2% downtime

            obs = {'C': cpu, 'M': mem, 'D': disk, 'N': net, 'H': hour}
            day_states.extend(infra_lang.encode_observation(obs)[1:])  # skip BOS per obs

        # Full day sequence: BOS + all observations
        docs.append([infra_lang.BOS] + day_states)

    print(f"Generated {len(docs)} day-sequences")
    print(f"Example decoded: {infra_lang.decode_sequence(docs[0][:16])}")

    # Create and train the atom
    atom = Atom(
        lang=infra_lang,
        n_embd=32,      # 32-dim embeddings
        n_head=4,        # 4 attention heads
        n_layer=2,       # 2 transformer layers
        block_size=16,   # see 16 tokens of context
    )
    print(f"\nAtom created: {atom.num_params:,} parameters")

    # Train
    num_steps = 300  # increase for better results
    print(f"\n--- Training for {num_steps} steps ---")
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        # Random window from the day
        start = random.randint(0, max(0, len(doc) - atom.block_size - 1))
        window = doc[start:start + atom.block_size + 1]
        if len(window) < 2:
            continue

        loss = atom.train_step(window)
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d} / {num_steps} | loss {loss:.4f}")

    # Test: predict and detect anomaly
    print("\n--- Anomaly Detection ---")

    # Normal observation (work hours, moderate CPU)
    normal = infra_lang.encode_observation({'C': 52, 'M': 43, 'D': 42, 'N': 1, 'H': 12})
    score_n, top3_n, actual_n = atom.anomaly_score(normal)
    print(f"\nNormal state: {infra_lang.decode_sequence(normal)}")
    print(f"  Anomaly score: {score_n:.2f}")
    print(f"  Actual last token: {actual_n}")
    print(f"  Model expected: {', '.join(f'{n}({p:.2f})' for n, p in top3_n)}")

    # Anomalous observation (midnight, high CPU — unusual)
    anomaly = infra_lang.encode_observation({'C': 95, 'M': 88, 'D': 42, 'N': 1, 'H': 3})
    score_a, top3_a, actual_a = atom.anomaly_score(anomaly)
    print(f"\nAnomalous state: {infra_lang.decode_sequence(anomaly)}")
    print(f"  Anomaly score: {score_a:.2f}")
    print(f"  Actual last token: {actual_a}")
    print(f"  Model expected: {', '.join(f'{n}({p:.2f})' for n, p in top3_a)}")

    if score_a > score_n:
        print(f"\n  ✓ Model correctly finds anomaly MORE surprising ({score_a:.2f} > {score_n:.2f})")
    else:
        print(f"\n  ✗ Need more training (try --num_steps 2000)")

    # Demo: The Pipe
    print("\n--- Pipe Demo (composition) ---")

    def observe(state_dict):
        """Stage 1: Encode observation."""
        return infra_lang.encode_observation(state_dict)

    def predict(tokens):
        """Stage 2: Get anomaly score."""
        score, top3, actual = atom.anomaly_score(tokens)
        return {'tokens': tokens, 'score': score, 'top3': top3, 'actual': actual}

    def decide(result):
        """Stage 3: Decide action based on score."""
        if result['score'] > 3.0:
            result['action'] = 'ALERT'
            result['message'] = f"Anomaly detected! Score: {result['score']:.1f}. Expected {result['top3'][0][0]}, got {result['actual']}"
        elif result['score'] > 2.0:
            result['action'] = 'WATCH'
            result['message'] = f"Unusual pattern. Score: {result['score']:.1f}"
        else:
            result['action'] = 'OK'
            result['message'] = 'Normal'
        return result

    def act(result):
        """Stage 4: Execute action (print for demo, webhook in prod)."""
        icons = {'ALERT': '🔴', 'WATCH': '🟡', 'OK': '🟢'}
        icon = icons.get(result['action'], '⚪')
        print(f"  {icon} [{result['action']}] {result['message']}")
        return result  # pass through for next pipe or logging

    pipe = Pipe()
    pipe.add('observe', observe)
    pipe.add('predict', predict)
    pipe.add('decide', decide)
    pipe.add('act', act)

    # Run pipe on normal state
    print("\nPipe run — normal (work hours, moderate load):")
    pipe.run({'C': 50, 'M': 40, 'D': 42, 'N': 1, 'H': 14})

    # Run pipe on anomalous state
    print("\nPipe run — anomaly (3am, maxed out):")
    pipe.run({'C': 98, 'M': 92, 'D': 42, 'N': 1, 'H': 3})

    # Run pipe on network down
    print("\nPipe run — network down during work:")
    pipe.run({'C': 45, 'M': 35, 'D': 42, 'N': 0, 'H': 10})

    print("\n" + "=" * 60)
    print("This is the atom. One model. One pipe.")
    print("Molecules = multiple atoms, different languages, chained.")
    print("Organisms = self-retraining loops.")
    print("=" * 60)
    print(f"\nTotal: {atom.num_params:,} params | {infra_lang.vocab_size} vocab | pure Python")
