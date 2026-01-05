#!/usr/bin/env python3
"""
Phase 9.3: QEC Training Data Generator

Generates training datasets for ML-based decoders by simulating
error patterns and extracting syndromes for various noise models.

Usage:
    python scripts/generate_qec_training_data.py --code surface --distance 5 \
        --noise-file noise_profile.json --num-samples 100000 --output train_data.npz
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class NoiseProfile:
    """Python representation of C++ NoiseProfile."""

    device_name: str = ""
    calibration_timestamp: str = ""
    num_qubits: int = 0
    t1_times_ns: List[float] = field(default_factory=list)
    t2_times_ns: List[float] = field(default_factory=list)
    single_gate_errors: List[float] = field(default_factory=list)
    two_qubit_errors: Dict[str, float] = field(default_factory=dict)
    readout_errors: List[float] = field(default_factory=list)
    correlated_errors: List[Dict] = field(default_factory=list)

    def avg_gate_error(self) -> float:
        if not self.single_gate_errors:
            return 0.001  # Default
        return np.mean(self.single_gate_errors)

    def avg_two_qubit_error(self) -> float:
        if not self.two_qubit_errors:
            return 0.01  # Default
        return np.mean(list(self.two_qubit_errors.values()))

    def avg_readout_error(self) -> float:
        if not self.readout_errors:
            return 0.01  # Default
        return np.mean(self.readout_errors)

    def effective_physical_error_rate(self) -> float:
        return self.avg_gate_error() + 2.0 * self.avg_two_qubit_error() + 0.5 * self.avg_readout_error()

    @classmethod
    def from_json(cls, path: str) -> "NoiseProfile":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            device_name=data.get("device_name", ""),
            calibration_timestamp=data.get("calibration_timestamp", ""),
            num_qubits=data.get("num_qubits", 0),
            t1_times_ns=data.get("t1_times_ns", []),
            t2_times_ns=data.get("t2_times_ns", []),
            single_gate_errors=data.get("single_gate_errors", []),
            two_qubit_errors=data.get("two_qubit_errors", {}),
            readout_errors=data.get("readout_errors", []),
            correlated_errors=data.get("correlated_errors", []),
        )

    @classmethod
    def default(cls, num_qubits: int, p_phys: float = 0.001) -> "NoiseProfile":
        """Create a default uniform noise profile."""
        return cls(
            device_name="synthetic",
            calibration_timestamp=datetime.now().isoformat(),
            num_qubits=num_qubits,
            t1_times_ns=[50000.0] * num_qubits,  # 50 µs
            t2_times_ns=[70000.0] * num_qubits,  # 70 µs
            single_gate_errors=[p_phys] * num_qubits,
            readout_errors=[0.01] * num_qubits,
        )


class StabilizerCode:
    """Base class for stabilizer codes."""

    def __init__(self, distance: int):
        self.distance = distance
        self._data_qubits: List[Tuple[int, int]] = []
        self._x_stabilizers: List[List[int]] = []
        self._z_stabilizers: List[List[int]] = []

    @property
    def num_data_qubits(self) -> int:
        return len(self._data_qubits)

    @property
    def num_x_stabilizers(self) -> int:
        return len(self._x_stabilizers)

    @property
    def num_z_stabilizers(self) -> int:
        return len(self._z_stabilizers)

    @property
    def syndrome_size(self) -> int:
        return self.num_x_stabilizers + self.num_z_stabilizers


class RepetitionCode(StabilizerCode):
    """Distance-d repetition code for bit-flip errors."""

    def __init__(self, distance: int):
        super().__init__(distance)

        # Data qubits on a line
        self._data_qubits = [(i, 0) for i in range(distance)]

        # Z stabilizers between adjacent data qubits
        self._z_stabilizers = [[i, i + 1] for i in range(distance - 1)]

        # No X stabilizers for repetition code (bit-flip only)
        self._x_stabilizers = []


class SurfaceCode(StabilizerCode):
    """Rotated surface code on a d x d grid."""

    def __init__(self, distance: int):
        super().__init__(distance)
        d = distance

        # Data qubits: d x d grid
        self._data_qubits = [(r, c) for r in range(d) for c in range(d)]
        self._qubit_index = {(r, c): i for i, (r, c) in enumerate(self._data_qubits)}

        # Build stabilizers
        self._build_stabilizers()

    def _build_stabilizers(self):
        """Build X and Z stabilizers for rotated surface code."""
        d = self.distance

        # Plaquette coordinates for rotated layout
        x_plaquettes = []
        z_plaquettes = []

        for r in range(d - 1):
            for c in range(d - 1):
                # Checkerboard pattern
                if (r + c) % 2 == 0:
                    x_plaquettes.append((r, c))
                else:
                    z_plaquettes.append((r, c))

        # Add boundary plaquettes
        for c in range(0, d - 1, 2):
            z_plaquettes.append((-1, c))  # Top boundary
            z_plaquettes.append((d - 1, c))  # Bottom boundary

        for r in range(1, d - 1, 2):
            x_plaquettes.append((r, -1))  # Left boundary
            x_plaquettes.append((r, d - 1))  # Right boundary

        # Convert plaquettes to qubit indices
        for pr, pc in x_plaquettes:
            neighbors = self._get_plaquette_qubits(pr, pc)
            if neighbors:
                self._x_stabilizers.append(neighbors)

        for pr, pc in z_plaquettes:
            neighbors = self._get_plaquette_qubits(pr, pc)
            if neighbors:
                self._z_stabilizers.append(neighbors)

    def _get_plaquette_qubits(self, pr: int, pc: int) -> List[int]:
        """Get data qubit indices around a plaquette."""
        d = self.distance
        neighbors = []

        # Four corners of plaquette
        corners = [(pr, pc), (pr, pc + 1), (pr + 1, pc), (pr + 1, pc + 1)]

        for r, c in corners:
            if 0 <= r < d and 0 <= c < d:
                neighbors.append(self._qubit_index[(r, c)])

        return neighbors


def create_code(code_type: str, distance: int) -> StabilizerCode:
    """Factory function for creating stabilizer codes."""
    if code_type.lower() == "repetition":
        return RepetitionCode(distance)
    elif code_type.lower() == "surface":
        return SurfaceCode(distance)
    else:
        raise ValueError(f"Unknown code type: {code_type}")


def sample_error(
    num_qubits: int,
    noise: NoiseProfile,
    rng: np.random.Generator,
    biased: bool = False,
) -> np.ndarray:
    """
    Sample a random Pauli error pattern.

    Returns:
        Array of shape (num_qubits,) with values in {0, 1, 2, 3}
        representing {I, X, Z, Y} errors.
    """
    p = noise.effective_physical_error_rate()

    if biased:
        # Biased noise: mostly Z errors
        p_x = p * 0.1
        p_z = p * 0.9
        p_y = p * 0.0
    else:
        # Depolarizing: equal X, Y, Z
        p_x = p / 3
        p_z = p / 3
        p_y = p / 3

    errors = np.zeros(num_qubits, dtype=np.int32)
    rand = rng.random(num_qubits)

    for q in range(num_qubits):
        r = rand[q]
        if r < p_x:
            errors[q] = 1  # X error
        elif r < p_x + p_z:
            errors[q] = 2  # Z error
        elif r < p_x + p_z + p_y:
            errors[q] = 3  # Y error
        # else: no error (I)

    return errors


def sample_correlated_error(
    num_qubits: int,
    noise: NoiseProfile,
    rng: np.random.Generator,
    correlation_prob: float = 0.1,
) -> np.ndarray:
    """Sample errors with correlations between neighboring qubits."""
    errors = sample_error(num_qubits, noise, rng)

    # Add correlated errors
    for ce in noise.correlated_errors:
        q1, q2 = ce.get("qubit_i", 0), ce.get("qubit_j", 1)
        if q1 < num_qubits and q2 < num_qubits:
            if rng.random() < correlation_prob and errors[q1] != 0:
                # Propagate error to neighbor
                errors[q2] = errors[q1]

    return errors


def extract_syndrome(code: StabilizerCode, error: np.ndarray) -> np.ndarray:
    """
    Extract syndrome from error pattern.

    For X stabilizers: detect Z and Y errors (anticommute with X)
    For Z stabilizers: detect X and Y errors (anticommute with Z)
    """
    x_syndrome = np.zeros(code.num_x_stabilizers, dtype=np.int32)
    z_syndrome = np.zeros(code.num_z_stabilizers, dtype=np.int32)

    # X stabilizers detect Z/Y errors
    for i, stab in enumerate(code._x_stabilizers):
        parity = 0
        for q in stab:
            if error[q] in (2, 3):  # Z or Y
                parity ^= 1
        x_syndrome[i] = parity

    # Z stabilizers detect X/Y errors
    for i, stab in enumerate(code._z_stabilizers):
        parity = 0
        for q in stab:
            if error[q] in (1, 3):  # X or Y
                parity ^= 1
        z_syndrome[i] = parity

    return np.concatenate([x_syndrome, z_syndrome])


def add_measurement_noise(
    syndrome: np.ndarray,
    noise: NoiseProfile,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add measurement (readout) noise to syndrome."""
    p_meas = noise.avg_readout_error()
    flip = rng.random(len(syndrome)) < p_meas
    return np.bitwise_xor(syndrome, flip.astype(np.int32))


def generate_samples(
    code: StabilizerCode,
    noise: NoiseProfile,
    num_samples: int,
    seed: int,
    add_measurement_error: bool = True,
    correlated: bool = False,
    biased: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training samples.

    Returns:
        syndromes: Array of shape (num_samples, syndrome_size)
        errors: Array of shape (num_samples, num_data_qubits)
    """
    rng = np.random.default_rng(seed)

    syndromes = []
    errors = []

    for i in range(num_samples):
        if (i + 1) % 10000 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples")

        # Sample error
        if correlated:
            error = sample_correlated_error(code.num_data_qubits, noise, rng)
        else:
            error = sample_error(code.num_data_qubits, noise, rng, biased=biased)

        # Extract syndrome
        syndrome = extract_syndrome(code, error)

        # Add measurement noise
        if add_measurement_error:
            syndrome = add_measurement_noise(syndrome, noise, rng)

        syndromes.append(syndrome)
        errors.append(error)

    return np.array(syndromes, dtype=np.int32), np.array(errors, dtype=np.int32)


def generate_multi_round_samples(
    code: StabilizerCode,
    noise: NoiseProfile,
    num_samples: int,
    num_rounds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate samples with multiple syndrome measurement rounds.

    Returns:
        syndromes: Array of shape (num_samples, num_rounds, syndrome_size)
        errors: Array of shape (num_samples, num_data_qubits)
    """
    rng = np.random.default_rng(seed)

    syndromes = []
    errors = []

    for i in range(num_samples):
        if (i + 1) % 10000 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples")

        # Sample cumulative error (errors accumulate over rounds)
        cumulative_error = np.zeros(code.num_data_qubits, dtype=np.int32)
        round_syndromes = []

        for _ in range(num_rounds):
            # New errors this round
            new_error = sample_error(code.num_data_qubits, noise, rng)

            # Combine errors (Pauli multiplication)
            for q in range(code.num_data_qubits):
                cumulative_error[q] = pauli_multiply(cumulative_error[q], new_error[q])

            # Extract syndrome for current error state
            syndrome = extract_syndrome(code, cumulative_error)
            syndrome = add_measurement_noise(syndrome, noise, rng)
            round_syndromes.append(syndrome)

        syndromes.append(np.stack(round_syndromes))
        errors.append(cumulative_error)

    return np.array(syndromes, dtype=np.int32), np.array(errors, dtype=np.int32)


def pauli_multiply(p1: int, p2: int) -> int:
    """Multiply two Pauli operators (ignoring phase)."""
    # 0=I, 1=X, 2=Z, 3=Y
    # I*X=X, I*Z=Z, I*Y=Y
    # X*X=I, X*Z=Y, X*Y=Z
    # Z*Z=I, Z*X=Y, Z*Y=X
    # Y*Y=I, Y*X=Z, Y*Z=X
    table = [
        [0, 1, 2, 3],  # I*
        [1, 0, 3, 2],  # X*
        [2, 3, 0, 1],  # Z*
        [3, 2, 1, 0],  # Y*
    ]
    return table[p1][p2]


def save_dataset(
    output_path: str,
    syndromes: np.ndarray,
    errors: np.ndarray,
    metadata: dict,
):
    """Save dataset to compressed NumPy format."""
    np.savez_compressed(
        output_path,
        syndromes=syndromes,
        errors=errors,
        metadata=json.dumps(metadata),
    )
    logger.info(f"Saved dataset to {output_path}")
    logger.info(f"  Syndromes shape: {syndromes.shape}")
    logger.info(f"  Errors shape: {errors.shape}")


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load dataset from compressed NumPy format."""
    data = np.load(path, allow_pickle=True)
    syndromes = data["syndromes"]
    errors = data["errors"]
    metadata = json.loads(str(data["metadata"]))
    return syndromes, errors, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate QEC training data for ML decoders"
    )
    parser.add_argument(
        "--code",
        type=str,
        default="surface",
        choices=["surface", "repetition"],
        help="Stabilizer code type",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=5,
        help="Code distance",
    )
    parser.add_argument(
        "--noise-file",
        type=str,
        default=None,
        help="Path to noise profile JSON (optional)",
    )
    parser.add_argument(
        "--p-phys",
        type=float,
        default=0.001,
        help="Physical error rate (if no noise file)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of syndrome measurement rounds (for temporal data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_data.npz",
        help="Output file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--correlated",
        action="store_true",
        help="Include correlated errors",
    )
    parser.add_argument(
        "--biased",
        action="store_true",
        help="Use biased noise (mostly Z errors)",
    )
    parser.add_argument(
        "--no-measurement-error",
        action="store_true",
        help="Disable measurement (readout) errors",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )

    args = parser.parse_args()

    logger.info(f"Generating training data for {args.code} code, d={args.distance}")

    # Create code
    code = create_code(args.code, args.distance)
    logger.info(f"Code has {code.num_data_qubits} data qubits, "
                f"syndrome size {code.syndrome_size}")

    # Load or create noise profile
    if args.noise_file:
        noise = NoiseProfile.from_json(args.noise_file)
        logger.info(f"Loaded noise profile from {args.noise_file}")
    else:
        noise = NoiseProfile.default(code.num_data_qubits, args.p_phys)
        logger.info(f"Using default noise profile with p_phys={args.p_phys}")

    logger.info(f"Effective physical error rate: {noise.effective_physical_error_rate():.4f}")

    # Generate samples
    if args.num_rounds > 1:
        logger.info(f"Generating {args.num_samples} multi-round samples "
                    f"({args.num_rounds} rounds)")
        syndromes, errors = generate_multi_round_samples(
            code, noise, args.num_samples, args.num_rounds, args.seed
        )
    else:
        logger.info(f"Generating {args.num_samples} single-round samples")
        syndromes, errors = generate_samples(
            code,
            noise,
            args.num_samples,
            args.seed,
            add_measurement_error=not args.no_measurement_error,
            correlated=args.correlated,
            biased=args.biased,
        )

    # Create metadata
    metadata = {
        "code_type": args.code,
        "code_distance": args.distance,
        "num_samples": args.num_samples,
        "num_rounds": args.num_rounds,
        "p_phys": args.p_phys,
        "correlated": args.correlated,
        "biased": args.biased,
        "measurement_error": not args.no_measurement_error,
        "creation_timestamp": datetime.now().isoformat(),
        "seed": args.seed,
    }

    # Split into train/validation
    if args.validation_split > 0:
        n_val = int(args.num_samples * args.validation_split)
        n_train = args.num_samples - n_val

        train_syndromes, val_syndromes = syndromes[:n_train], syndromes[n_train:]
        train_errors, val_errors = errors[:n_train], errors[n_train:]

        # Save training set
        train_meta = {**metadata, "split": "train", "num_samples": n_train}
        save_dataset(args.output, train_syndromes, train_errors, train_meta)

        # Save validation set
        val_path = Path(args.output).stem + "_val.npz"
        val_meta = {**metadata, "split": "validation", "num_samples": n_val}
        save_dataset(val_path, val_syndromes, val_errors, val_meta)
    else:
        save_dataset(args.output, syndromes, errors, metadata)

    logger.info("Done!")


if __name__ == "__main__":
    main()
