# LRET Coq + QuantumLib Verification

Independent formal verification of LRET mathematical foundations using Coq and QuantumLib.

## Dependencies

- Coq 8.17+
- QuantumLib: `opam install coq-quantumlib`

## Building

```bash
make
```

## Theorems Verified

### KrausCompleteness.v
- `depolarizing_kraus_complete`: Σ Kᵢ†Kᵢ = I for depolarizing channel
- `amplitude_damping_kraus_complete`: Σ Kᵢ†Kᵢ = I for amplitude damping
- `phase_damping_kraus_complete`: Σ Kᵢ†Kᵢ = I for phase damping
- `bit_flip_kraus_complete`: Σ Kᵢ†Kᵢ = I for bit flip
- `phase_flip_kraus_complete`: Σ Kᵢ†Kᵢ = I for phase flip

### ChoiIsomorphism.v
- `choi_gate_evolution_2`: (U†⊗U)·vec(ρ) = vec(UρU†) for 1-qubit gates
- `gate_evolution_density_matrix`: Unitary evolution preserves density matrix properties
- `kron_gate_unitary`: Tensor product of unitaries is unitary

### TracePreservation.v
- `trace_cyclic`: Tr(AB) = Tr(BA)
- `llstar_trace_eq`: Tr(LL†) = Tr(L†L) — LRET core identity
- `llstar_psd`: LL† is positive semidefinite
- `cptp_trace_preserving_2op`: CPTP maps preserve trace
- `lret_density_matrix_valid`: LRET representation gives valid density matrix

## Relationship to Lean 4 Verification

These theorems are independently verified from the Lean 4 / Mathlib proofs in `lean/`.
Independent verification in two formal systems provides publication-grade confidence.
