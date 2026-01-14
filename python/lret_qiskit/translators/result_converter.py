"""Convert LRET simulation results to Qiskit Result format."""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

from qiskit.result import Result
from qiskit.providers.jobstatus import JobStatus

__all__ = ["ResultConverter"]


class ResultConverter:
    """Converts LRET simulation output to Qiskit Result objects."""

    def __init__(self, backend_name: str, backend_version: str = "0.1.0"):
        """Initialize converter.
        
        Args:
            backend_name: Name of the backend for result metadata.
            backend_version: Version string for result metadata.
        """
        self.backend_name = backend_name
        self.backend_version = backend_version

    def convert(
        self,
        lret_results: List[Dict[str, Any]],
        circuits: List,
        job_id: str,
        shots: int,
    ) -> Result:
        """Convert LRET results to Qiskit Result.
        
        Args:
            lret_results: List of LRET simulation result dicts.
            circuits: Original Qiskit circuits (for metadata).
            job_id: Job identifier.
            shots: Number of shots requested.
        
        Returns:
            Qiskit Result object.
        """
        experiments = []
        total_time = 0.0

        for circuit, lret_result in zip(circuits, lret_results):
            exp_data = self._convert_single(lret_result, circuit, shots)
            experiments.append(exp_data)
            total_time += lret_result.get("execution_time_ms", 0) / 1000.0

        return Result.from_dict({
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "qobj_id": str(uuid4()),
            "job_id": job_id,
            "success": all(e["success"] for e in experiments),
            "results": experiments,
            "status": JobStatus.DONE.name,
            "time_taken": total_time,
        })

    def _convert_single(
        self,
        lret_result: Dict[str, Any],
        circuit,
        shots: int,
    ) -> Dict[str, Any]:
        """Convert a single LRET result to experiment result dict.
        
        Args:
            lret_result: Single LRET simulation result.
            circuit: Original Qiskit circuit.
            shots: Number of shots.
        
        Returns:
            Experiment result dict for Qiskit Result.
        """
        success = lret_result.get("status") == "success"
        
        data: Dict[str, Any] = {}

        # Extract counts from samples if available
        if "samples" in lret_result and lret_result["samples"]:
            samples = lret_result["samples"]
            counts = self._samples_to_counts(samples, circuit.num_clbits or circuit.num_qubits)
            data["counts"] = counts
        
        # Extract expectation values
        if "expectation_values" in lret_result:
            data["expectation_values"] = lret_result["expectation_values"]
        
        # Extract probabilities if available
        if "probabilities" in lret_result:
            data["probabilities"] = lret_result["probabilities"]

        # Include final rank for diagnostics
        if "final_rank" in lret_result:
            data["final_rank"] = lret_result["final_rank"]

        return {
            "shots": shots,
            "success": success,
            "data": data,
            "header": {
                "name": circuit.name,
                "n_qubits": circuit.num_qubits,
                "execution_time_ms": lret_result.get("execution_time_ms", 0),
            },
        }

    def _samples_to_counts(
        self,
        samples: List[int],
        num_bits: int,
    ) -> Dict[str, int]:
        """Convert sample list to counts dictionary.
        
        Args:
            samples: List of measurement outcomes as integers.
            num_bits: Number of bits for formatting.
        
        Returns:
            Dictionary mapping bitstrings to counts.
        """
        counts: Dict[str, int] = {}
        for sample in samples:
            bitstring = format(sample, f"0{num_bits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
