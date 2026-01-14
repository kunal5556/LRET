"""Job wrapper for LRET Qiskit backend.

Executes Qiskit circuits through the LRET simulator core (qlret.api).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from qiskit.providers import JobError, JobV1
from qiskit.providers.jobstatus import JobStatus
from qiskit.result import Result

from ..translators import CircuitTranslator, ResultConverter

# Import LRET simulation API
try:
    from qlret.api import simulate_json, QLRETError
    _HAS_QLRET = True
except ImportError:
    _HAS_QLRET = False
    QLRETError = RuntimeError  # type: ignore

__all__ = ["LRETJob"]


class LRETJob(JobV1):
    """Executes Qiskit circuits using the LRET low-rank simulator.
    
    This job translates Qiskit circuits to LRET JSON format, runs them
    through the native LRET backend, and converts results back to
    Qiskit Result format.
    """

    def __init__(
        self,
        backend,
        job_id: str,
        circuits: List,
        options,
    ) -> None:
        """Initialize LRET job.
        
        Args:
            backend: The LRETBackend instance.
            job_id: Unique job identifier.
            circuits: List of Qiskit QuantumCircuit objects.
            options: Backend options (shots, epsilon, etc.).
        """
        super().__init__(backend, job_id)
        self._circuits: List = list(circuits)
        self._options = options
        self._status = JobStatus.INITIALIZING
        self._result: Optional[Result] = None
        self._error: Optional[Exception] = None
        self._lret_results: List[Dict[str, Any]] = []

    def submit(self) -> None:
        """Submit the job for execution.
        
        Runs synchronously since LRET is a local simulator.
        """
        if self._status in {JobStatus.RUNNING, JobStatus.DONE}:
            return

        if not _HAS_QLRET:
            self._status = JobStatus.ERROR
            self._error = ImportError(
                "LRET simulator not available. Install with: pip install qlret"
            )
            raise JobError(str(self._error))

        self._status = JobStatus.RUNNING
        try:
            self._result = self._execute()
            self._status = JobStatus.DONE
        except Exception as exc:
            self._error = exc
            self._status = JobStatus.ERROR
            raise JobError(str(exc)) from exc

    def result(self, timeout: Optional[float] = None) -> Result:
        """Get job results.
        
        Args:
            timeout: Not used (job runs synchronously).
        
        Returns:
            Qiskit Result object.
        
        Raises:
            JobError: If job failed or has not completed.
        """
        if self._status == JobStatus.ERROR:
            raise JobError(str(self._error))
        if self._result is None:
            raise JobError("Job has not completed execution")
        return self._result

    def status(self) -> JobStatus:
        """Return current job status."""
        return self._status

    def cancel(self) -> bool:
        """Attempt to cancel the job.
        
        Returns:
            True if cancellation was successful.
        """
        if self._status in {JobStatus.DONE, JobStatus.ERROR}:
            return False
        self._status = JobStatus.CANCELLED
        return True

    def error_message(self) -> Optional[str]:
        """Return error message if job failed."""
        if self._error:
            return str(self._error)
        return None

    def _execute(self) -> Result:
        """Execute circuits through LRET and build Result.
        
        Returns:
            Qiskit Result object with simulation outcomes.
        """
        shots = int(self._options.shots)
        epsilon = float(self._options.epsilon)

        # Initialize translator with job parameters
        translator = CircuitTranslator(epsilon=epsilon, shots=shots)
        
        # Initialize result converter
        converter = ResultConverter(
            backend_name=self._backend.name,
            backend_version=getattr(self._backend, "_backend_version", "0.1.0"),
        )

        lret_results: List[Dict[str, Any]] = []

        for circuit in self._circuits:
            # Translate Qiskit circuit to LRET JSON
            circuit_json = translator.translate(circuit)
            
            # Run through LRET simulator
            try:
                result = simulate_json(circuit_json, export_state=False)
                lret_results.append(result)
            except QLRETError as exc:
                # Wrap LRET errors
                lret_results.append({
                    "status": "error",
                    "error": str(exc),
                    "execution_time_ms": 0,
                })

        self._lret_results = lret_results

        # Convert to Qiskit Result format
        return converter.convert(
            lret_results=lret_results,
            circuits=self._circuits,
            job_id=self.job_id(),
            shots=shots,
        )
