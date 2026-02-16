"""Tests for the Atom transformer — init, forward, train, save/load, anomaly scoring."""

import math
import json
import pytest
from kiri.core.value import Value
from kiri.core.atom import Atom
from kiri.core.language import StateLanguage


class TestAtomInit:
    def test_param_count(self, small_atom):
        assert small_atom.num_params > 0

    def test_vocab_matches_lang(self, small_atom, small_lang):
        assert small_atom.vocab_size == small_lang.vocab_size

    def test_config_stored(self, small_atom):
        assert small_atom.n_embd == 16
        assert small_atom.n_head == 2
        assert small_atom.n_layer == 1
        assert small_atom.block_size == 8


class TestAtomForward:
    def test_forward_returns_logits(self, small_atom):
        keys = [[] for _ in range(small_atom.n_layer)]
        vals = [[] for _ in range(small_atom.n_layer)]
        logits = small_atom.forward(0, 0, keys, vals)
        assert len(logits) == small_atom.vocab_size
        assert all(isinstance(v, Value) for v in logits)

    def test_softmax_sums_to_one(self, small_atom):
        keys = [[] for _ in range(small_atom.n_layer)]
        vals = [[] for _ in range(small_atom.n_layer)]
        logits = small_atom.forward(0, 0, keys, vals)
        probs = small_atom._softmax(logits)
        total = sum(p.data for p in probs)
        assert abs(total - 1.0) < 1e-5


class TestAtomTraining:
    def test_train_step_returns_loss(self, small_atom, small_lang):
        obs = {'A': 3.0, 'B': 50.0}
        tokens = small_lang.encode_observation(obs)
        loss = small_atom.train_step(tokens, lr=0.01)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self, small_lang):
        atom = Atom(small_lang, n_embd=16, n_head=2, n_layer=1, block_size=8)
        obs = {'A': 3.0, 'B': 50.0}
        tokens = small_lang.encode_observation(obs)

        first_loss = atom.train_step(tokens, lr=0.01)
        for _ in range(29):
            atom.train_step(tokens, lr=0.01)
        last_loss = atom.train_step(tokens, lr=0.01)

        assert last_loss < first_loss


class TestAtomSaveLoad:
    def test_save_load_roundtrip(self, small_atom, small_lang, tmp_path):
        path = str(tmp_path / 'weights.json')
        # Train a few steps so weights diverge from init
        tokens = small_lang.encode_observation({'A': 3.0, 'B': 50.0})
        for _ in range(5):
            small_atom.train_step(tokens, lr=0.01)

        small_atom.save(path)

        loaded = Atom(small_lang, n_embd=16, n_head=2, n_layer=1, block_size=8)
        loaded.load_weights(path)

        # Compare a sample weight
        orig_val = small_atom.sd['wte'][0][0].data
        load_val = loaded.sd['wte'][0][0].data
        assert abs(orig_val - load_val) < 1e-10

    def test_step_count_persists(self, small_atom, small_lang, tmp_path):
        tokens = small_lang.encode_observation({'A': 3.0, 'B': 50.0})
        for _ in range(5):
            small_atom.train_step(tokens, lr=0.01)
        path = str(tmp_path / 'weights.json')
        small_atom.save(path)

        loaded = Atom(small_lang, n_embd=16, n_head=2, n_layer=1, block_size=8)
        loaded.load_weights(path)
        assert loaded.step_count == 5


class TestAnomalyScore:
    def test_normal_scores_lower_than_anomalous(self, small_lang):
        """Train on one pattern, verify it scores lower than an unseen pattern."""
        atom = Atom(small_lang, n_embd=16, n_head=2, n_layer=1, block_size=8)
        normal_obs = {'A': 3.0, 'B': 50.0}
        normal_tokens = small_lang.encode_observation(normal_obs)

        for _ in range(50):
            atom.train_step(normal_tokens, lr=0.01)

        anomalous_obs = {'A': 9.0, 'B': 5.0}
        anomalous_tokens = small_lang.encode_observation(anomalous_obs)

        normal_score, _, _ = atom.anomaly_score(normal_tokens)
        anomalous_score, _, _ = atom.anomaly_score(anomalous_tokens)

        assert normal_score < anomalous_score

    def test_short_sequence_returns_zero(self, small_atom):
        score, top3, actual = small_atom.anomaly_score([0])
        assert score == 0.0
