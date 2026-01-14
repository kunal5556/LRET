"""BackendV2 implementation for LRET mixed-state simulator."""

from __future__ import annotations

from typing import Iterable, List, Sequence
from uuid import uuid4

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import (
    CXGate,
    CZGate,
    HGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    UGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.circuit.measure import Measure
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import InstructionProperties, Target

from .lret_job import LRETJob


class LRETBackend(BackendV2):
    """Qiskit backend wrapper around the LRET simulator core."""

    version = 2

    def __init__(
        self,
        name: str,
        description: str,
        epsilon: float,
        provider=None,
        num_qubits: int = 20,
    ) -> None:
        # Set epsilon BEFORE calling super().__init__ because
        # BackendV2.__init__ calls _default_options() which needs it
        self._epsilon = float(epsilon)
        self._num_qubits = int(num_qubits)
        
        super().__init__(provider=provider, name=name, description=description)
        
        self._target = self._build_target(self._num_qubits)
        self._backend_version = "0.1.0"
        self._max_circuits = None

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self):
        return self._max_circuits

    @classmethod
    def _default_options(cls) -> Options:
        # Default options - epsilon will be overridden per-instance
        return Options(shots=1024, epsilon=1e-4, max_parallel=1)

    def run(self, run_input: QuantumCircuit | Sequence[QuantumCircuit], **options):
        circuits: List[QuantumCircuit] = (
            list(run_input) if isinstance(run_input, Iterable) and not isinstance(run_input, QuantumCircuit) else [run_input]  # type: ignore[arg-type]
        )

        # Build options dict with backend's epsilon as default
        resolved = {
            "shots": self.options.shots,
            "epsilon": self._epsilon,  # Use this backend's epsilon
            "max_parallel": self.options.max_parallel,
        }
        resolved.update(options)
        
        # Create Options object
        resolved_opts = Options(**resolved)

        job = LRETJob(self, str(uuid4()), circuits, resolved_opts)
        job.submit()
        return job

    def _build_target(self, num_qubits: int) -> Target:
        target = Target(num_qubits=num_qubits, description="LRET mixed-state simulator")

        single_qubit_ops = [
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            SGate(),
            SdgGate(),
            TGate(),
            TdgGate(),
            RXGate(0),
            RYGate(0),
            RZGate(0),
            PhaseGate(0),
            UGate(0, 0, 0),
        ]

        for op in single_qubit_ops:
            properties = {(q,): InstructionProperties() for q in range(num_qubits)}
            target.add_instruction(op, properties)

        two_qubit_ops = [CXGate(), CZGate()]
        for op in two_qubit_ops:
            properties = {
                (q0, q1): InstructionProperties()
                for q0 in range(num_qubits)
                for q1 in range(num_qubits)
                if q0 != q1
            }
            target.add_instruction(op, properties)

        measure = Measure()
        meas_props = {(q,): InstructionProperties() for q in range(num_qubits)}
        target.add_instruction(measure, meas_props)

        return target
