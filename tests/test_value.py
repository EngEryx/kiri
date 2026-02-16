"""Tests for the scalar autograd engine (Value class)."""

import math
import pytest
from kiri.core.value import Value


class TestValueOps:
    def test_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_mul(self):
        a = Value(3.0)
        b = Value(4.0)
        c = a * b
        assert c.data == 12.0

    def test_pow(self):
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0

    def test_neg(self):
        a = Value(5.0)
        c = -a
        assert c.data == -5.0

    def test_sub(self):
        a = Value(7.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 4.0

    def test_div(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert abs(c.data - 2.0) < 1e-6

    def test_log(self):
        a = Value(math.e)
        c = a.log()
        assert abs(c.data - 1.0) < 1e-6

    def test_exp(self):
        a = Value(1.0)
        c = a.exp()
        assert abs(c.data - math.e) < 1e-6

    def test_relu_positive(self):
        a = Value(3.0)
        c = a.relu()
        assert c.data == 3.0

    def test_relu_negative(self):
        a = Value(-3.0)
        c = a.relu()
        assert c.data == 0.0

    def test_radd(self):
        a = Value(2.0)
        c = 5 + a
        assert c.data == 7.0

    def test_rmul(self):
        a = Value(3.0)
        c = 4 * a
        assert c.data == 12.0


class TestBackward:
    def test_add_backward(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_mul_backward(self):
        a = Value(3.0)
        b = Value(4.0)
        c = a * b
        c.backward()
        assert a.grad == 4.0  # dc/da = b
        assert b.grad == 3.0  # dc/db = a

    def test_chain_rule(self):
        # f = (a * b) + (a * c)  => df/da = b + c
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        d = a * b + a * c
        d.backward()
        assert a.grad == 7.0  # b + c = 3 + 4
        assert b.grad == 2.0  # a
        assert c.grad == 2.0  # a

    def test_pow_backward(self):
        a = Value(3.0)
        c = a ** 2
        c.backward()
        assert abs(a.grad - 6.0) < 1e-6  # d(a^2)/da = 2a

    def test_relu_backward(self):
        a = Value(3.0)
        c = a.relu()
        c.backward()
        assert a.grad == 1.0

        b = Value(-3.0)
        d = b.relu()
        d.backward()
        assert b.grad == 0.0

    def test_complex_expression(self):
        # Neuron-like: y = relu(w*x + b)
        x = Value(2.0)
        w = Value(3.0)
        b = Value(-1.0)
        y = (w * x + b).relu()
        y.backward()
        assert y.data == 5.0
        assert w.grad == 2.0  # dy/dw = x (relu passes through since output > 0)
        assert x.grad == 3.0  # dy/dx = w
        assert b.grad == 1.0
