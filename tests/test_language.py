"""Tests for StateLanguage — tokenization and bucket encoding."""

import pytest
from kiri.core.language import StateLanguage


class TestStateLanguage:
    def test_vocab_size(self, small_lang):
        # BOS + 3 (A buckets) + 5 (B buckets) = 9
        assert small_lang.vocab_size == 9

    def test_bos_is_zero(self, small_lang):
        assert small_lang.BOS == 0
        assert small_lang.tokens[0] == '<BOS>'

    def test_encode_value_boundaries(self, small_lang):
        # A: (0, 10, 3) => buckets A0, A1, A2
        # 0 => A0, 3.3 => A0, 3.4 => A1, 6.7 => A2, 10 => A2
        assert small_lang.decode_token(small_lang.encode_value('A', 0)) == 'A0'
        assert small_lang.decode_token(small_lang.encode_value('A', 10)) == 'A2'
        assert small_lang.decode_token(small_lang.encode_value('A', 5)) == 'A1'

    def test_encode_value_clamps(self, small_lang):
        # Values below min or above max should clamp to first/last bucket
        assert small_lang.decode_token(small_lang.encode_value('A', -5)) == 'A0'
        assert small_lang.decode_token(small_lang.encode_value('A', 100)) == 'A2'

    def test_encode_observation(self, small_lang):
        obs = {'A': 5.0, 'B': 50.0}
        tokens = small_lang.encode_observation(obs)
        assert tokens[0] == small_lang.BOS
        assert len(tokens) == 3  # BOS + A + B

    def test_encode_decode_roundtrip(self, small_lang):
        obs = {'A': 5.0, 'B': 80.0}
        tokens = small_lang.encode_observation(obs)
        decoded = small_lang.decode_sequence(tokens)
        assert '<BOS>' in decoded
        assert 'A' in decoded
        assert 'B' in decoded

    def test_prefix_map(self, small_lang):
        assert len(small_lang.prefix_map['A']) == 3
        assert len(small_lang.prefix_map['B']) == 5

    def test_stoi_itos_consistency(self, small_lang):
        for token, idx in small_lang.stoi.items():
            assert small_lang.itos[idx] == token

    def test_save_load_roundtrip(self, small_lang, tmp_path):
        path = str(tmp_path / 'lang.json')
        small_lang.save(path)
        loaded = StateLanguage.load(path)
        assert loaded.name == small_lang.name
        assert loaded.vocab_size == small_lang.vocab_size
        assert loaded.schema == small_lang.schema

    def test_schema_order_preserved(self):
        schema = {'X': (0, 1, 2), 'Y': (0, 10, 3), 'Z': (0, 100, 4)}
        lang = StateLanguage('ordered', schema)
        obs = {'X': 0.5, 'Y': 5.0, 'Z': 50.0}
        tokens = lang.encode_observation(obs)
        decoded = lang.decode_sequence(tokens)
        parts = decoded.split()
        # Order after BOS should be X, Y, Z
        assert parts[1].startswith('X')
        assert parts[2].startswith('Y')
        assert parts[3].startswith('Z')
