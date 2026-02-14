"""Molecule — MoE transformer that combines all atom domains with explanation generation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class MoleculeLanguage:
    """Unified vocabulary across all domains with explanation words."""

    def __init__(self, domain_schemas, actions, control_tokens, explanation_words):
        """
        domain_schemas: dict of {prefix: (min_val, max_val, num_buckets)}
            Prefixed per domain: p.C, r.I, d.T, PS, H, etc.
        actions: list of action strings ['ok', 'alert', ...]
        control_tokens: list like ['<SEP>', '<EXP>', '<END>']
        explanation_words: list of explanation vocabulary words
        """
        self.tokens = ['<BOS>']
        self.stoi = {}
        self.itos = {}
        self.schema = domain_schemas
        self.prefix_map = {}

        # Domain tokens from schema
        for prefix, (lo, hi, buckets) in domain_schemas.items():
            bucket_tokens = []
            for b in range(buckets):
                token = f"{prefix}{b}"
                bucket_tokens.append(len(self.tokens))
                self.tokens.append(token)
            self.prefix_map[prefix] = bucket_tokens

        # Action tokens
        self.action_tokens = []
        for action in actions:
            token = f'A:{action}'
            self.action_tokens.append(len(self.tokens))
            self.tokens.append(token)

        # Control tokens
        self.control_ids = {}
        for ct in control_tokens:
            self.control_ids[ct] = len(self.tokens)
            self.tokens.append(ct)

        # Explanation words
        self.explanation_ids = []
        for word in explanation_words:
            self.explanation_ids.append(len(self.tokens))
            self.tokens.append(word)

        # Build lookup tables
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for i, t in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
        self.BOS = 0

        # Precompute the set of valid generation tokens (explanation + <END>)
        self.gen_token_ids = set(self.explanation_ids)
        if '<END>' in self.control_ids:
            self.gen_token_ids.add(self.control_ids['<END>'])

    def encode_value(self, prefix, value):
        lo, hi, buckets = self.schema[prefix]
        bucket = int((value - lo) / (hi - lo) * buckets)
        bucket = max(0, min(buckets - 1, bucket))
        token = f"{prefix}{bucket}"
        return self.stoi[token]

    def encode_observation(self, obs_dict):
        """Encode {prefix: value} into token sequence (no BOS)."""
        tokens = []
        for prefix in self.schema:
            if prefix in obs_dict:
                tokens.append(self.encode_value(prefix, obs_dict[prefix]))
        return tokens

    def decode_token(self, token_id):
        return self.itos.get(token_id, '?')

    def decode_sequence(self, token_ids):
        return ' '.join(self.decode_token(t) for t in token_ids)

    def save(self, path):
        import json
        data = {
            'schema': {k: list(v) for k, v in self.schema.items()},
            'actions': [self.decode_token(i) for i in self.action_tokens],
            'control_tokens': list(self.control_ids.keys()),
            'explanation_words': [self.decode_token(i) for i in self.explanation_ids],
            'tokens': self.tokens,
            'vocab_size': self.vocab_size,
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            data = json.load(f)
        schema = {k: tuple(v) for k, v in data['schema'].items()}
        actions = [a.replace('A:', '') for a in data['actions']]
        return cls(schema, actions, data['control_tokens'], data['explanation_words'])


class Molecule(nn.Module):
    """MoE transformer for cross-domain anomaly diagnosis with explanations."""

    def __init__(self, lang, n_embd=48, n_head=4, n_layer=3,
                 block_size=32, n_experts=4, top_k=2, ffn_dim=96):
        super().__init__()
        self.lang = lang
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.ffn_dim = ffn_dim
        self.head_dim = n_embd // n_head
        self.vocab_size = lang.vocab_size
        self.device = _get_device()

        self.wte = nn.Embedding(self.vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            layer = nn.ModuleDict({
                'wq': nn.Linear(n_embd, n_embd, bias=False),
                'wk': nn.Linear(n_embd, n_embd, bias=False),
                'wv': nn.Linear(n_embd, n_embd, bias=False),
                'wo': nn.Linear(n_embd, n_embd, bias=False),
                'gate': nn.Linear(n_embd, n_experts, bias=False),
            })
            for e in range(n_experts):
                layer[f'e{e}_f1'] = nn.Linear(n_embd, ffn_dim, bias=False)
                layer[f'e{e}_f2'] = nn.Linear(ffn_dim, n_embd, bias=False)
            self.layers.append(layer)

        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)
        self.to(self.device)

        self.num_params = sum(p.numel() for p in self.parameters())

    def _rmsnorm(self, x):
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + 1e-5)

    def _moe_ffn(self, x, layer):
        """MoE FFN: gate selects top-k of n_experts, weighted sum."""
        B, T, D = x.shape
        gate_logits = layer['gate'](x)                          # (B, T, n_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)             # (B, T, n_experts)

        topk_vals, topk_idx = gate_probs.topk(self.top_k, dim=-1)  # (B, T, top_k)
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # Compute all experts
        expert_outs = []
        for e in range(self.n_experts):
            h = layer[f'e{e}_f1'](x)
            h = F.relu(h)
            h = layer[f'e{e}_f2'](h)
            expert_outs.append(h)
        expert_stack = torch.stack(expert_outs, dim=2)          # (B, T, n_experts, D)

        # Gather top-k and weighted sum
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        selected = expert_stack.gather(2, idx_expanded)         # (B, T, top_k, D)
        out = (selected * topk_weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)

        # Load-balancing aux loss: encourage uniform expert usage
        fraction = gate_probs.mean(dim=(0, 1))                 # (n_experts,)
        aux_loss = self.n_experts * (fraction * fraction).sum()

        return out, aux_loss

    def forward(self, idx):
        """Forward pass. idx: (B, T) -> (logits: (B, T, V), aux_loss: scalar)"""
        B, T = idx.shape
        pos = torch.arange(T, device=self.device)

        x = self.wte(idx) + self.wpe(pos)
        x = self._rmsnorm(x)

        mask = torch.triu(torch.ones(T, T, device=self.device), diagonal=1).bool()
        total_aux = torch.tensor(0.0, device=self.device)

        for layer in self.layers:
            xr = x
            x = self._rmsnorm(x)
            q = layer['wq'](x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = layer['wk'](x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = layer['wv'](x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = attn.masked_fill(mask[:T, :T], float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.n_embd)
            x = layer['wo'](out) + xr

            xr = x
            x = self._rmsnorm(x)
            moe_out, aux = self._moe_ffn(x, layer)
            x = moe_out + xr
            total_aux = total_aux + aux

        logits = self.lm_head(x)
        aux_loss = 0.01 * total_aux / self.n_layer
        return logits, aux_loss

    def train_step(self, token_sequences, lr=0.01):
        """Train on a batch of sequences. Returns average loss."""
        self.train()

        if not hasattr(self, '_optimizer'):
            self._optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, betas=(0.85, 0.99)
            )

        max_len = min(self.block_size + 1, max(len(s) for s in token_sequences))
        padded = [s[:max_len] + [0] * (max_len - len(s[:max_len])) for s in token_sequences]
        idx = torch.tensor(padded, device=self.device)

        inputs = idx[:, :-1]
        targets = idx[:, 1:]

        logits, aux_loss = self.forward(inputs)
        ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1))
        loss = ce_loss + aux_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict_next(self, token_sequence, temperature=0.5):
        """Predict probability distribution over next token."""
        self.eval()
        idx = torch.tensor([token_sequence[-self.block_size:]], device=self.device)
        logits, _ = self.forward(idx)
        logits = logits[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        return [(self.lang.decode_token(i), p.item()) for i, p in enumerate(probs)]

    @torch.no_grad()
    def anomaly_score(self, token_sequence):
        """Average anomaly score across all tokens."""
        self.eval()
        if len(token_sequence) < 2:
            return 0.0, []

        idx = torch.tensor([token_sequence[-self.block_size:]], device=self.device)
        logits, _ = self.forward(idx[:, :-1])
        targets = idx[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        T = targets.shape[1]
        per_token = []
        for t in range(T):
            target_id = targets[0, t].item()
            score = -log_probs[0, t, target_id].item()
            prob = math.exp(-score)
            per_token.append((self.lang.decode_token(target_id), score, prob))

        avg = sum(s for _, s, _ in per_token) / len(per_token)
        return avg, per_token

    @torch.no_grad()
    def predict_action(self, obs_tokens):
        """Given observation tokens, predict the best action."""
        self.eval()
        seq = [self.lang.BOS] + list(obs_tokens)
        preds = self.predict_next(seq)
        action_preds = [(name, prob) for name, prob in preds if name.startswith('A:')]
        action_preds.sort(key=lambda x: -x[1])
        if action_preds:
            best = action_preds[0][0].replace('A:', '')
            return best, action_preds
        return 'ok', action_preds

    @torch.no_grad()
    def explain(self, obs_tokens, action=None, max_tokens=10):
        """Generate a short diagnostic explanation."""
        self.eval()
        seq = [self.lang.BOS] + list(obs_tokens)

        # Predict action if not given
        if action is None:
            action, _ = self.predict_action(obs_tokens)

        action_token = f'A:{action}'
        if action_token in self.lang.stoi:
            seq.append(self.lang.stoi[action_token])

        # Add <EXP> control token
        if '<EXP>' in self.lang.control_ids:
            seq.append(self.lang.control_ids['<EXP>'])

        end_id = self.lang.control_ids.get('<END>')
        gen_ids = self.lang.gen_token_ids

        words = []
        for _ in range(max_tokens):
            idx = torch.tensor([seq[-self.block_size:]], device=self.device)
            logits, _ = self.forward(idx)
            logits = logits[0, -1] / 0.5

            # Mask to only explanation words + <END>
            mask = torch.full_like(logits, float('-inf'))
            for tid in gen_ids:
                mask[tid] = 0.0
            logits = logits + mask

            probs = F.softmax(logits, dim=-1)
            next_id = probs.argmax().item()

            if next_id == end_id:
                break

            words.append(self.lang.decode_token(next_id))
            seq.append(next_id)

        return ' '.join(words)

    def save(self, path):
        torch.save({
            'config': {
                'n_embd': self.n_embd, 'n_head': self.n_head,
                'n_layer': self.n_layer, 'block_size': self.block_size,
                'n_experts': self.n_experts, 'top_k': self.top_k,
                'ffn_dim': self.ffn_dim,
            },
            'state_dict': self.state_dict(),
        }, path)

    def load_weights(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(data['state_dict'])
