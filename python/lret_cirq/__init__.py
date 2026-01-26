"""LRET Cirq Integration.

This package provides a Cirq-compatible simulator interface backed by LRET's
low-rank quantum simulation engine.

Example:
    >>> import cirq
    >>> from lret_cirq import LRETSimulator
    >>> 
    >>> q0, q1 = cirq.LineQubit.range(2)
    >>> circuit = cirq.Circuit(
    ...     cirq.H(q0),
    ...     cirq.CNOT(q0, q1),
    ...     cirq.measure(q0, q1, key='result')
    ... )
    >>> 
    >>> sim = LRETSimulator(epsilon=1e-4)
    >>> result = sim.run(circuit, repetitions=1000)
    >>> print(result.histogram(key='result'))
"""

from .lret_simulator import LRETSimulator
from .translators.circuit_translator import CircuitTranslator, TranslationError
from .translators.result_converter import ResultConverter

__all__ = [
    "LRETSimulator",
    "CircuitTranslator",
    "TranslationError",
    "ResultConverter",
]

__version__ = "0.1.0"
