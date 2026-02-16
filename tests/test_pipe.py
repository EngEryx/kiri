"""Tests for the Pipe composition engine."""

import pytest
from kiri.core.pipe import Pipe


class TestPipe:
    def test_empty_pipe_returns_input(self):
        pipe = Pipe()
        result = pipe.run(42)
        assert result == 42

    def test_single_stage(self):
        pipe = Pipe()
        pipe.add('double', lambda x: x * 2)
        assert pipe.run(5) == 10

    def test_chained_stages(self):
        pipe = Pipe()
        pipe.add('add1', lambda x: x + 1)
        pipe.add('double', lambda x: x * 2)
        result = pipe.run(3)
        assert result == 8  # (3+1)*2

    def test_add_returns_self(self):
        pipe = Pipe()
        ret = pipe.add('noop', lambda x: x)
        assert ret is pipe

    def test_fluent_chaining(self):
        result = (
            Pipe()
            .add('a', lambda x: x + 1)
            .add('b', lambda x: x * 3)
            .run(2)
        )
        assert result == 9  # (2+1)*3

    def test_log_records_stages(self):
        pipe = Pipe()
        pipe.add('inc', lambda x: x + 1)
        pipe.add('sq', lambda x: x ** 2)
        pipe.run(4)
        assert len(pipe.log) == 2
        assert pipe.log[0]['stage'] == 'inc'
        assert pipe.log[1]['stage'] == 'sq'
        assert 'time' in pipe.log[0]

    def test_log_truncates_long_output(self):
        pipe = Pipe()
        pipe.add('big', lambda x: 'a' * 500)
        pipe.run(0)
        assert len(pipe.log[0]['output']) <= 200

    def test_loop_single_iteration(self):
        pipe = Pipe()
        pipe.add('inc', lambda x: x + 1)
        result = pipe.loop(0, iterations=1)
        assert result == 1

    def test_loop_multiple_iterations(self):
        pipe = Pipe()
        pipe.add('inc', lambda x: x + 1)
        result = pipe.loop(0, iterations=5)
        assert result == 5

    def test_loop_with_zero_delay(self):
        pipe = Pipe()
        pipe.add('noop', lambda x: x)
        result = pipe.loop(42, iterations=1, delay=0)
        assert result == 42

    def test_loop_with_positive_delay(self):
        pipe = Pipe()
        pipe.add('inc', lambda x: x + 1)
        result = pipe.loop(0, iterations=2, delay=0.001)
        assert result == 2

    def test_dict_data_flow(self):
        pipe = Pipe()
        pipe.add('enrich', lambda d: {**d, 'score': d['value'] * 2})
        pipe.add('label', lambda d: {**d, 'label': 'high' if d['score'] > 5 else 'low'})
        result = pipe.run({'value': 4})
        assert result == {'value': 4, 'score': 8, 'label': 'high'}
