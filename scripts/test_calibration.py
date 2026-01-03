#!/usr/bin/env python3
"""Lightweight sanity checks for calibration scripts."""

import math
from fit_depolarizing import estimate_depolarizing
from fit_t1_t2 import _fit_exponential


def test_estimate_depolarizing() -> None:
    p = estimate_depolarizing([0.99, 0.985, 0.98])
    assert 0.01 <= p <= 0.02


def test_fit_exponential() -> None:
    # Synthetic decay: y = exp(-t/50) for t in [0..100]
    times = [0.0, 25.0, 50.0, 75.0, 100.0]
    vals = [math.exp(-t / 50.0) for t in times]
    T = _fit_exponential(times, vals)
    assert 45.0 <= T <= 55.0


def run_all() -> None:
    test_estimate_depolarizing()
    test_fit_exponential()
    print("Calibration tests passed")


if __name__ == "__main__":
    run_all()
