"""Tests for Molecule — MoE transformer with explanations."""

import pytest
import json
import tempfile
from pathlib import Path

torch = pytest.importorskip('torch')

from kiri.core.molecule import MoleculeLanguage, Molecule
from kiri.atoms.molecule.config import (
    PULSE_TOKENS, RHYTHM_TOKENS, DRIFT_TOKENS, SCORE_TOKENS, TEMPORAL_TOKENS,
    ACTIONS, CONTROL_TOKENS, EXPLANATION_WORDS,
)
from kiri.atoms.molecule.train import make_molecule_language, build_sequences
from kiri.atoms.molecule.collect import generate_synthetic


@pytest.fixture
def lang():
    return make_molecule_language()


@pytest.fixture
def model(lang):
    return Molecule(lang, n_embd=48, n_head=4, n_layer=3,
                    block_size=32, n_experts=4, top_k=2, ffn_dim=96)


# --- MoleculeLanguage tests ---

class TestMoleculeLanguage:
    def test_vocab_size(self, lang):
        # BOS + pulse(42) + rhythm(29) + drift(26) + scores(15) + temporal(13)
        # + actions(4) + control(3) + explanation(~50)
        assert lang.vocab_size > 100
        assert lang.vocab_size < 250

    def test_bos_token(self, lang):
        assert lang.BOS == 0
        assert lang.decode_token(0) == '<BOS>'

    def test_domain_prefix_encoding(self, lang):
        # Pulse CPU at 50% -> p.C5 (bucket 5 of 10)
        tid = lang.encode_value('p.C', 50)
        assert lang.decode_token(tid) == 'p.C5'

    def test_rhythm_encoding(self, lang):
        tid = lang.encode_value('r.I', 1800)  # mid-range idle
        name = lang.decode_token(tid)
        assert name.startswith('r.I')

    def test_score_encoding(self, lang):
        tid = lang.encode_value('PS', 10)  # mid anomaly score
        name = lang.decode_token(tid)
        assert name.startswith('PS')

    def test_action_tokens_exist(self, lang):
        for action in ACTIONS:
            token = f'A:{action}'
            assert token in lang.stoi

    def test_control_tokens_exist(self, lang):
        for ct in CONTROL_TOKENS:
            assert ct in lang.stoi
            assert ct in lang.control_ids

    def test_explanation_words_exist(self, lang):
        for word in EXPLANATION_WORDS:
            assert word in lang.stoi

    def test_encode_observation(self, lang):
        obs = {'p.C': 50, 'p.M': 70, 'r.I': 200, 'PS': 5.0, 'H': 14}
        tokens = lang.encode_observation(obs)
        assert len(tokens) > 0
        # All tokens should be valid
        for t in tokens:
            assert 0 <= t < lang.vocab_size

    def test_no_prefix_collision(self, lang):
        # p.C and d.C should produce different tokens
        pulse_c = lang.encode_value('p.C', 50)
        drift_c = lang.encode_value('d.C', 10)
        assert pulse_c != drift_c
        assert lang.decode_token(pulse_c).startswith('p.C')
        assert lang.decode_token(drift_c).startswith('d.C')

    def test_save_load_roundtrip(self, lang):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        lang.save(path)
        loaded = MoleculeLanguage.load(path)
        assert loaded.vocab_size == lang.vocab_size
        assert loaded.tokens == lang.tokens
        Path(path).unlink()


# --- Molecule model tests ---

class TestMolecule:
    def test_forward_shape(self, model, lang):
        obs = {'p.C': 50, 'p.M': 70, 'p.D': 45, 'p.S': 10, 'p.L': 2, 'p.N': 1,
               'r.I': 200, 'r.A': 15, 'r.H': 14, 'r.W': 2,
               'd.T': 3, 'd.C': 2, 'd.S': 1, 'd.H': 14, 'd.W': 1,
               'PS': 2.0, 'RS': 1.5, 'DS': 1.0,
               'H': 14, 'W': 2}
        tokens = [lang.BOS] + lang.encode_observation(obs)
        idx = torch.tensor([tokens], device=model.device)
        logits, aux_loss = model.forward(idx)
        assert logits.shape == (1, len(tokens), lang.vocab_size)
        assert aux_loss.item() >= 0

    def test_param_count(self, model):
        assert 100_000 < model.num_params < 250_000

    def test_train_loss_decreases(self, lang):
        model = Molecule(lang, n_embd=48, n_head=4, n_layer=3,
                         block_size=32, n_experts=4, top_k=2, ffn_dim=96)
        obs = generate_synthetic(n_samples=100)
        seqs = build_sequences(obs, lang)
        assert len(seqs) > 0

        # Train a few steps
        import random
        losses = []
        for step in range(20):
            batch = [seqs[random.randint(0, len(seqs) - 1)] for _ in range(8)]
            loss = model.train_step(batch, lr=0.01)
            losses.append(loss)

        # Loss should decrease (comparing first 5 avg vs last 5 avg)
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early

    def test_predict_action(self, model, lang):
        obs = {'p.C': 90, 'p.M': 80, 'p.D': 50, 'p.S': 40, 'p.L': 12, 'p.N': 1,
               'r.I': 100, 'r.A': 30, 'r.H': 10, 'r.W': 1,
               'd.T': 2, 'd.C': 1, 'd.S': 0, 'd.H': 10, 'd.W': 1,
               'PS': 8.0, 'RS': 1.5, 'DS': 0.9,
               'H': 10, 'W': 1}
        tokens = lang.encode_observation(obs)
        action, probs = model.predict_action(tokens)
        assert action in ACTIONS
        assert len(probs) > 0

    def test_explain_generates_valid_tokens(self, model, lang):
        obs = {'p.C': 50, 'p.M': 40, 'p.D': 45, 'p.S': 10, 'p.L': 2, 'p.N': 1,
               'r.I': 300, 'r.A': 15, 'r.H': 14, 'r.W': 2,
               'd.T': 3, 'd.C': 2, 'd.S': 1, 'd.H': 14, 'd.W': 2,
               'PS': 1.0, 'RS': 1.0, 'DS': 0.8,
               'H': 14, 'W': 2}
        tokens = lang.encode_observation(obs)
        explanation = model.explain(tokens)
        # Explanation should be a string
        assert isinstance(explanation, str)
        # Each word should be in the explanation vocabulary
        if explanation:
            for word in explanation.split():
                assert word in EXPLANATION_WORDS

    def test_anomaly_score(self, model, lang):
        obs = {'p.C': 50, 'p.M': 40, 'p.D': 45, 'p.S': 10, 'p.L': 2, 'p.N': 1,
               'r.I': 300, 'r.A': 15, 'r.H': 14, 'r.W': 2,
               'd.T': 3, 'd.C': 2, 'd.S': 1, 'd.H': 14, 'd.W': 2,
               'PS': 1.0, 'RS': 1.0, 'DS': 0.8,
               'H': 14, 'W': 2}
        tokens = [lang.BOS] + lang.encode_observation(obs)
        avg, per_token = model.anomaly_score(tokens)
        assert isinstance(avg, float)
        assert avg >= 0
        assert len(per_token) > 0

    def test_save_load_roundtrip(self, lang):
        model = Molecule(lang, n_embd=48, n_head=4, n_layer=3,
                         block_size=32, n_experts=4, top_k=2, ffn_dim=96)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            weights_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            lang_path = f.name

        model.save(weights_path)
        lang.save(lang_path)

        # Load into new model
        loaded_lang = MoleculeLanguage.load(lang_path)
        loaded_model = Molecule(loaded_lang, n_embd=48, n_head=4, n_layer=3,
                                block_size=32, n_experts=4, top_k=2, ffn_dim=96)
        loaded_model.load_weights(weights_path)

        # Same predictions
        obs = {'p.C': 50, 'p.M': 40, 'PS': 2.0, 'H': 14}
        tokens = [lang.BOS] + lang.encode_observation(obs)
        idx = torch.tensor([tokens], device=model.device)

        model.eval()
        loaded_model.eval()
        logits1, _ = model.forward(idx)
        logits2, _ = loaded_model.forward(idx.to(loaded_model.device))
        assert torch.allclose(logits1.cpu(), logits2.cpu(), atol=1e-5)

        Path(weights_path).unlink()
        Path(lang_path).unlink()


# --- Synthetic data tests ---

class TestSyntheticData:
    def test_generate_synthetic(self):
        obs = generate_synthetic(n_samples=100)
        assert len(obs) == 100
        for o in obs:
            assert 'scenario' in o
            assert 'action' in o
            assert 'explanation' in o
            assert 'pulse' in o
            assert 'rhythm' in o
            assert 'drift' in o
            assert 'scores' in o

    def test_build_sequences(self, lang):
        obs = generate_synthetic(n_samples=50)
        seqs = build_sequences(obs, lang)
        assert len(seqs) > 0
        for seq in seqs:
            assert seq[0] == lang.BOS
            assert len(seq) <= 32  # block_size
            # Last token should be <END>
            assert lang.decode_token(seq[-1]) == '<END>'
