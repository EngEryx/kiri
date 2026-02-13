"""Quick demo of the atom concept with minimal params for speed."""
import os, math, random, json, time

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data; self.grad = 0; self._children = children; self._local_grads = local_grads
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
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self); self.grad = 1
        for v in reversed(topo):
            for c, lg in zip(v._children, v._local_grads): c.grad += lg * v.grad

# State language: quantize values into token buckets
class Lang:
    def __init__(self, schema):
        self.schema = schema
        self.tokens = ['<BOS>']
        for prefix, (lo, hi, buckets) in schema.items():
            for b in range(buckets): self.tokens.append(f"{prefix}{b}")
        self.vocab_size = len(self.tokens)
        self.stoi = {t:i for i,t in enumerate(self.tokens)}
        self.itos = {i:t for i,t in enumerate(self.tokens)}
    def encode_val(self, prefix, value):
        lo, hi, buckets = self.schema[prefix]
        b = max(0, min(buckets-1, int((value-lo)/(hi-lo)*buckets)))
        return self.stoi[f"{prefix}{b}"]
    def encode(self, obs):
        return [0] + [self.encode_val(p, obs[p]) for p in self.schema if p in obs]
    def decode(self, ids):
        return ' '.join(self.itos.get(i,'?') for i in ids)

# Tiny model: n_embd=16, 1 layer, 2 heads
def make_model(vocab_size, n_embd=16, n_head=2, n_layer=1, block_size=8):
    mat = lambda r,c,s=0.08: [[Value(random.gauss(0,s)) for _ in range(c)] for _ in range(r)]
    sd = {'wte':mat(vocab_size,n_embd), 'wpe':mat(block_size,n_embd), 'lm':mat(vocab_size,n_embd)}
    for i in range(n_layer):
        for k in ['wq','wk','wv','wo']: sd[f'{i}.{k}'] = mat(n_embd,n_embd)
        sd[f'{i}.f1'] = mat(4*n_embd, n_embd); sd[f'{i}.f2'] = mat(n_embd, 4*n_embd)
    params = [p for m in sd.values() for r in m for p in r]
    return sd, params, n_embd, n_head, n_layer, block_size

def linear(x,w): return [sum(wi*xi for wi,xi in zip(wo,x)) for wo in w]
def softmax(lg):
    mx = max(v.data for v in lg); ex = [(v-mx).exp() for v in lg]; t=sum(ex); return [e/t for e in ex]
def rmsnorm(x):
    ms = sum(xi*xi for xi in x)/len(x); s=(ms+1e-5)**-0.5; return [xi*s for xi in x]

def forward(sd, tok, pos, keys, vals, ne, nh, nl, bs):
    hd = ne//nh
    x = [t+p for t,p in zip(sd['wte'][tok], sd['wpe'][pos%bs])]
    x = rmsnorm(x)
    for li in range(nl):
        xr=x; x=rmsnorm(x)
        q=linear(x,sd[f'{li}.wq']); k=linear(x,sd[f'{li}.wk']); v=linear(x,sd[f'{li}.wv'])
        keys[li].append(k); vals[li].append(v)
        xa=[]
        for h in range(nh):
            hs=h*hd; qh=q[hs:hs+hd]
            kh=[ki[hs:hs+hd] for ki in keys[li]]; vh=[vi[hs:hs+hd] for vi in vals[li]]
            al=[sum(qh[j]*kh[t][j] for j in range(hd))/hd**0.5 for t in range(len(kh))]
            aw=softmax(al)
            xa.extend([sum(aw[t]*vh[t][j] for t in range(len(vh))) for j in range(hd)])
        x=linear(xa,sd[f'{li}.wo']); x=[a+b for a,b in zip(x,xr)]
        xr=x; x=rmsnorm(x); x=linear(x,sd[f'{li}.f1']); x=[xi.relu() for xi in x]
        x=linear(x,sd[f'{li}.f2']); x=[a+b for a,b in zip(x,xr)]
    return linear(x, sd['lm'])

# ============================================================
print("="*60)
print("ATOM — State Prediction from Scratch")
print("="*60)

# Define state language
lang = Lang({
    'C': (0, 100, 5),  # CPU: 5 buckets (20% each)
    'M': (0, 100, 5),  # Mem: 5 buckets
    'N': (0, 1, 2),    # Net: up/down
    'H': (0, 24, 4),   # Hour: 4 buckets (6hr each)
})
print(f"\nVocab: {lang.vocab_size} tokens → {lang.tokens}")

# Synthetic data: normal patterns
random.seed(42)
data = []
for day in range(300):
    for hour_block in range(4):
        hour = hour_block * 6
        work = 1 if hour_block in [1,2] else 0  # blocks 1,2 = work hours
        cpu = random.gauss(60 if work else 20, 12)
        mem = random.gauss(50 if work else 25, 10)
        net = 1 if random.random() > 0.03 else 0
        data.append(lang.encode({'C':cpu, 'M':mem, 'N':net, 'H':hour}))

print(f"Training sequences: {len(data)}")
print(f"Example: {lang.decode(data[0])} → {lang.decode(data[1])}")

# Build model
random.seed(42)
sd, params, ne, nh, nl, bs = make_model(lang.vocab_size, n_embd=16, n_head=2, n_layer=1, block_size=8)
print(f"Model: {len(params):,} params")

# Train: feed pairs of consecutive observations
adam_m = [0.0]*len(params)
adam_v = [0.0]*len(params)
lr = 0.01

print(f"\n--- Training (100 steps) ---")
for step in range(100):
    # Pick two consecutive observations as a sequence
    idx = random.randint(0, len(data)-2)
    seq = data[idx] + data[idx+1]  # concat two observations
    n = min(bs, len(seq)-1)

    keys = [[] for _ in range(nl)]
    vals = [[] for _ in range(nl)]
    losses = []
    for pos in range(n):
        logits = forward(sd, seq[pos], pos, keys, vals, ne, nh, nl, bs)
        probs = softmax(logits)
        losses.append(-probs[seq[pos+1]].log())
    loss = (1/n) * sum(losses)
    loss.backward()

    # Adam
    lr_t = lr * max(0.1, 1 - step/100)
    for i,p in enumerate(params):
        adam_m[i] = 0.85*adam_m[i] + 0.15*p.grad
        adam_v[i] = 0.99*adam_v[i] + 0.01*p.grad**2
        mh = adam_m[i]/(1-0.85**(step+1))
        vh = adam_v[i]/(1-0.99**(step+1))
        p.data -= lr_t * mh/(vh**0.5+1e-8)
        p.grad = 0

    if (step+1) % 20 == 0:
        print(f"  step {step+1:3d} | loss {loss.data:.4f}")

# Test anomaly detection
print(f"\n--- Anomaly Detection ---")

def anomaly_score(seq):
    if len(seq) < 2: return 0, [], ''
    ctx, actual = seq[:-1], seq[-1]
    keys = [[] for _ in range(nl)]; vals = [[] for _ in range(nl)]
    for pos in range(len(ctx)):
        logits = forward(sd, ctx[pos], pos, keys, vals, ne, nh, nl, bs)
    probs = softmax(logits)
    pred_sorted = sorted([(lang.itos[i], p.data) for i,p in enumerate(probs)], key=lambda x:-x[1])
    actual_name = lang.itos.get(actual, '?')
    actual_prob = next((p for n,p in pred_sorted if n==actual_name), 0.001)
    return -math.log(max(actual_prob,1e-10)), pred_sorted[:3], actual_name

# Normal: work hours, moderate CPU
normal = lang.encode({'C':55, 'M':48, 'N':1, 'H':12})
s1, t1, a1 = anomaly_score(normal)
print(f"\n🟢 Normal (work hours, CPU~55%)")
print(f"   State: {lang.decode(normal)}")
print(f"   Score: {s1:.2f} | Actual: {a1} | Expected: {', '.join(f'{n}({p:.2f})' for n,p in t1)}")

# Anomaly: 3am, maxed CPU
anomaly = lang.encode({'C':95, 'M':90, 'N':1, 'H':3})
s2, t2, a2 = anomaly_score(anomaly)
print(f"\n🔴 Anomaly (3am, CPU maxed)")
print(f"   State: {lang.decode(anomaly)}")
print(f"   Score: {s2:.2f} | Actual: {a2} | Expected: {', '.join(f'{n}({p:.2f})' for n,p in t2)}")

# Pipe demo
print(f"\n--- The Pipe (composition) ---")
print(f"   observe → predict → decide → act → log → [loop]")

def pipe(obs_dict):
    tokens = lang.encode(obs_dict)
    score, top3, actual = anomaly_score(tokens)
    if score > 3.0:
        action = '🔴 ALERT'
    elif score > 2.0:
        action = '🟡 WATCH'
    else:
        action = '🟢 OK'
    state_str = lang.decode(tokens)
    print(f"   {action} | score={score:.1f} | {state_str}")
    return {'action': action, 'score': score, 'state': state_str, 'obs': obs_dict}

# Simulate a day
print(f"\n   Simulating 24 hours...")
for h in [0, 6, 9, 12, 15, 18, 21, 3]:
    work = 6 <= h <= 18
    cpu = random.gauss(50 if work else 15, 10)
    mem = random.gauss(45 if work else 20, 8)
    if h == 3: cpu, mem = 95, 88  # inject anomaly at 3am
    result = pipe({'C': cpu, 'M': mem, 'N': 1, 'H': h})

# The key insight
print(f"\n{'='*60}")
print("THE MOLECULE BLUEPRINT:")
print("  Atom A (infra)  ──→ detects server anomaly")
print("  Atom B (behavior)──→ detects you stopped working")
print("  Atom C (tasks)   ──→ detects scope creep pattern")
print("  Atom D (nudge)   ──→ generates reminder text")
print("")
print("  PIPE: A.output + B.output → decision → D.generate → Telegram")
print("  LOOP: result of nudge → logged → feeds B on next cycle")
print("")
print("  Each atom: ~4K params. All four: ~16K params total.")
print("  Runs on Mac Mini forever. Zero API costs.")
print("  Trains on YOUR data. Gets smarter over time.")
print(f"{'='*60}")
