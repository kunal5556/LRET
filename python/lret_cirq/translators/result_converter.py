"""Result Converter: LRET → Cirq Result.

Converts LRET JSON results to Cirq Result objects, handling measurement
key organization and qubit index mapping.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple

try:
    import cirq
except ImportError:
    raise ImportError(
        "Cirq is required for lret_cirq. Install with: pip install cirq>=1.3.0"
    )

__all__ = ["ResultConverter"]


class ResultConverter:
    """
    Converts LRET results to Cirq Result format.
    
    Handles:
    - Sample array conversion and reshaping
    - Qubit index mapping (int → cirq.Qid)
    - Measurement key organization
    - Multiple measurement support
    
    Example:
        >>> converter = ResultConverter()
        >>> measurements = converter.convert(lret_result, circuit, qubit_map, measurement_info)
    """
    
    def __init__(self):
        """Initialize converter."""
        pass
    
    def convert(
        self,
        lret_result: Dict[str, Any],
        qubit_map: Dict[cirq.Qid, int],
        measurement_keys: List[str],
        measurement_qubits: Dict[str, List[cirq.Qid]],
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        """
        Convert LRET result to Cirq measurements dict.
        
        Args:
            lret_result: LRET simulation output dict
            qubit_map: Mapping from cirq.Qid to int index
            measurement_keys: List of measurement keys in order
            measurement_qubits: Mapping from key to list of measured qubits
            repetitions: Number of shots requested
            
        Returns:
            Dict mapping measurement keys to 3D sample arrays 
            (shape: repetitions × 1 × num_qubits) for Cirq's records format
        """
        # Extract samples from LRET result (2D: reps × qubits)
        samples = self._extract_samples(lret_result, len(qubit_map), repetitions)
        
        # If no explicit measurements, return all qubits as 'result'
        if not measurement_keys:
            # Convert to 3D: (reps, 1, qubits) for Cirq records format
            return {"result": np.expand_dims(samples, axis=1)}
        
        # Build measurements dict organized by key
        measurements_dict: Dict[str, np.ndarray] = {}
        
        for key in measurement_keys:
            if key not in measurement_qubits:
                continue
            
            qubits = measurement_qubits[key]
            qubit_indices = [qubit_map[q] for q in qubits]
            
            # Extract columns for these qubits
            key_samples = samples[:, qubit_indices]
            # Convert to 3D: (reps, 1, qubits) for Cirq records format
            measurements_dict[key] = np.expand_dims(key_samples, axis=1)
        
        return measurements_dict
    
    def _extract_samples(
        self,
        lret_result: Dict[str, Any],
        num_qubits: int,
        repetitions: int,
    ) -> np.ndarray:
        """
        Extract measurement samples from LRET result.
        
        LRET may return samples in different formats:
        - "samples": List of bitstrings or integers
        - "counts": Dict mapping bitstring to count
        
        Returns:
            numpy array of shape (repetitions, num_qubits) with 0/1 values
        """
        # Check for direct samples
        if "samples" in lret_result:
            samples = lret_result["samples"]
            
            if isinstance(samples, np.ndarray):
                if samples.ndim == 1:
                    # Integer samples - convert to bit arrays
                    return self._integers_to_bits(samples, num_qubits)
                else:
                    return samples.astype(np.int32)
            
            if isinstance(samples, list):
                if len(samples) == 0:
                    return np.zeros((repetitions, num_qubits), dtype=np.int32)
                
                if isinstance(samples[0], (list, np.ndarray)):
                    # Already bit arrays
                    return np.array(samples, dtype=np.int32)
                
                if isinstance(samples[0], (int, np.integer)):
                    # Integer samples - convert to bit arrays
                    return self._integers_to_bits(np.array(samples), num_qubits)
                
                if isinstance(samples[0], str):
                    # Bitstring samples
                    return self._bitstrings_to_array(samples, num_qubits)
        
        # Check for counts dict
        if "counts" in lret_result:
            counts = lret_result["counts"]
            return self._counts_to_samples(counts, num_qubits, repetitions)
        
        # Generate samples from probabilities if available
        if "probabilities" in lret_result:
            probs = lret_result["probabilities"]
            return self._probabilities_to_samples(probs, num_qubits, repetitions)
        
        # Fallback: return zeros (all measurements give |0⟩)
        return np.zeros((repetitions, num_qubits), dtype=np.int32)
    
    def _integers_to_bits(
        self,
        integers: np.ndarray,
        num_qubits: int,
    ) -> np.ndarray:
        """
        Convert integer measurements to bit arrays.
        
        LRET uses little-endian bit ordering: qubit 0 is the LSB.
        This matches Cirq's convention where the first measured qubit
        corresponds to column 0 in the output array.
        
        Args:
            integers: Array of measurement outcomes as integers
            num_qubits: Number of qubits (bits per integer)
            
        Returns:
            Array of shape (len(integers), num_qubits) with 0/1 values
            where column j contains the measurement result for qubit j.
        """
        n = len(integers)
        result = np.zeros((n, num_qubits), dtype=np.int32)
        
        for i, val in enumerate(integers):
            for j in range(num_qubits):
                # Bit j (qubit j) goes to column j
                result[i, j] = (int(val) >> j) & 1
        
        return result
    
    def _bitstrings_to_array(
        self,
        bitstrings: List[str],
        num_qubits: int,
    ) -> np.ndarray:
        """
        Convert bitstring samples to numpy array.
        
        Args:
            bitstrings: List of bitstrings like ["00", "11", "01"]
            num_qubits: Expected number of qubits
            
        Returns:
            Array of shape (len(bitstrings), num_qubits) with 0/1 values
        """
        n = len(bitstrings)
        result = np.zeros((n, num_qubits), dtype=np.int32)
        
        for i, bs in enumerate(bitstrings):
            # Pad or truncate to num_qubits
            bs = bs.zfill(num_qubits)[-num_qubits:]
            for j, bit in enumerate(bs):
                result[i, j] = int(bit)
        
        return result
    
    def _counts_to_samples(
        self,
        counts: Dict[str, int],
        num_qubits: int,
        repetitions: int,
    ) -> np.ndarray:
        """
        Expand counts dict to individual samples.
        
        Args:
            counts: Dict mapping bitstring/int to count
            num_qubits: Number of qubits
            repetitions: Target number of samples
            
        Returns:
            Array of shape (repetitions, num_qubits)
        """
        samples = []
        
        for key, count in counts.items():
            # Convert key to bits
            if isinstance(key, str):
                bits = [int(b) for b in key.zfill(num_qubits)[-num_qubits:]]
            else:
                bits = [(int(key) >> (num_qubits - 1 - j)) & 1 for j in range(num_qubits)]
            
            # Add count copies
            for _ in range(count):
                samples.append(bits)
        
        result = np.array(samples, dtype=np.int32)
        
        # Pad or truncate to repetitions
        if len(result) < repetitions:
            # Repeat to fill
            repeats = (repetitions // len(result)) + 1
            result = np.tile(result, (repeats, 1))[:repetitions]
        elif len(result) > repetitions:
            result = result[:repetitions]
        
        return result
    
    def _probabilities_to_samples(
        self,
        probs: List[float],
        num_qubits: int,
        repetitions: int,
    ) -> np.ndarray:
        """
        Sample from probability distribution.
        
        Args:
            probs: Probability for each computational basis state
            num_qubits: Number of qubits
            repetitions: Number of samples to generate
            
        Returns:
            Array of shape (repetitions, num_qubits)
        """
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize
        
        # Sample indices
        indices = np.random.choice(len(probs), size=repetitions, p=probs)
        
        # Convert to bits
        return self._integers_to_bits(indices, num_qubits)
    
    def build_cirq_result(
        self,
        measurements: Dict[str, np.ndarray],
        params: cirq.ParamResolver = None,
    ) -> cirq.ResultDict:
        """
        Build a Cirq ResultDict from measurements.
        
        Args:
            measurements: Dict mapping keys to sample arrays
            params: Optional parameter resolver used
            
        Returns:
            cirq.ResultDict object
        """
        return cirq.ResultDict(
            params=params or cirq.ParamResolver(),
            measurements=measurements,
        )
