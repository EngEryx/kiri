"""Shared test fixtures for KIRI core engine tests."""

import sys
from pathlib import Path

# Ensure kiri package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from kiri.core.language import StateLanguage
from kiri.core.atom import Atom


@pytest.fixture
def small_schema():
    """Minimal schema for fast tests."""
    return {
        'A': (0, 10, 3),
        'B': (0, 100, 5),
    }


@pytest.fixture
def small_lang(small_schema):
    """StateLanguage with a small vocabulary (9 tokens: BOS + 3 + 5)."""
    return StateLanguage('test', small_schema)


@pytest.fixture
def small_atom(small_lang):
    """Tiny Atom for fast test iterations."""
    return Atom(small_lang, n_embd=16, n_head=2, n_layer=1, block_size=8)
