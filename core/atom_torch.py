"""AtomTorch — same Atom architecture on PyTorch with MPS acceleration."""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class AtomTorch(nn.Module):
    """A tiny GPT trained on state sequences. PyTorch + MPS accelerated."""

    def __init__(self, lang, n_embd=32, n_head=4, n_layer=2, block_size=16):
        super().__init__()
        self.lang = lang
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.head_dim = n_embd // n_head
        self.vocab_size = lang.vocab_size
        self.device = _get_device()

        self.wte = nn.Embedding(self.vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(nn.ModuleDict({
                'wq': nn.Linear(n_embd, n_embd, bias=False),
                'wk': nn.Linear(n_embd, n_embd, bias=False),
                'wv': nn.Linear(n_embd, n_embd, bias=False),
                'wo': nn.Linear(n_embd, n_embd, bias=False),
                'f1': nn.Linear(n_embd, 4 * n_embd, bias=False),
                'f2': nn.Linear(4 * n_embd, n_embd, bias=False),
            }))

        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)
        self.to(self.device)

        self.num_params = sum(p.numel() for p in self.parameters())

    def _rmsnorm(self, x):
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + 1e-5)

    def forward(self, idx):
        """Forward pass on a batch of sequences. idx: (B, T) -> logits: (B, T, V)"""
        B, T = idx.shape
        pos = torch.arange(T, device=self.device)

        x = self.wte(idx) + self.wpe(pos)
        x = self._rmsnorm(x)

        mask = torch.triu(torch.ones(T, T, device=self.device), diagonal=1).bool()

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
            x = layer['f1'](x)
            x = F.relu(x)
            x = layer['f2'](x) + xr

        return self.lm_head(x)

    def train_step(self, token_sequences, lr=0.01):
        """Train on a batch of sequences. Returns average loss."""
        self.train()

        if not hasattr(self, '_optimizer'):
            self._optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, betas=(0.85, 0.99)
            )

        # token_sequences: list of lists of ints
        max_len = min(self.block_size + 1, max(len(s) for s in token_sequences))
        padded = [s[:max_len] + [0] * (max_len - len(s[:max_len])) for s in token_sequences]
        idx = torch.tensor(padded, device=self.device)

        inputs = idx[:, :-1]
        targets = idx[:, 1:]

        logits = self.forward(inputs)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1))

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict_next(self, token_sequence, temperature=0.5):
        """Predict probability distribution over next token."""
        self.eval()
        idx = torch.tensor([token_sequence[-self.block_size:]], device=self.device)
        logits = self.forward(idx)
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
        logits = self.forward(idx[:, :-1])
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

    def save(self, path):
        """Save model weights."""
        torch.save({
            'config': {
                'n_embd': self.n_embd, 'n_head': self.n_head,
                'n_layer': self.n_layer, 'block_size': self.block_size,
            },
            'state_dict': self.state_dict(),
        }, path)

    def load_weights(self, path):
        """Load model weights."""
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(data['state_dict'])
