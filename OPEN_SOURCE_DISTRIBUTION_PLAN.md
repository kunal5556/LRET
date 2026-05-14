# LRET Open-Source Distribution Plan

This document is the implementation plan for distributing the LRET (Low-Rank Entanglement Tracking) quantum simulator across the major open-source quantum software ecosystems. It is **strategy-and-execution** focused: every target lists what to build, how to ship it, what the acceptance criteria are, and what effort and reward to expect.

Built from the empirical evidence in [results/pub_pennylane_reg/](results/pub_pennylane_reg/) and [results/pub_small_parallel_r3/](results/pub_small_parallel_r3/): LRET is **6–90× faster than `default.mixed` at N≥8** on noisy variational circuits, and is **mathematically equivalent across all six parallel modes** (SEQUENTIAL, ROW, COLUMN, BATCH, HYBRID, LAYER_PARALLEL — Frobenius error 0).

---

## Contents

1. [Strategic principle](#strategic-principle)
2. [Cross-cutting prerequisites (do these first)](#cross-cutting-prerequisites-do-these-first)
3. [Priority ranking — at a glance](#priority-ranking--at-a-glance)
4. [Tier 1 — Highest returns, do first](#tier-1--highest-returns-do-first)
   1. [PennyLane plugin gallery](#11-pennylane-plugin-gallery-xanadu)
   2. [Qiskit BackendV2 adapter](#12-qiskit-backendv2-adapter-ibm)
   3. [Cirq simulator adapter](#13-cirq-simulator-adapter-google)
   4. [Braket local-simulator plugin](#14-braket-local-simulator-plugin-aws)
5. [Tier 2 — High strategic value, medium effort](#tier-2--high-strategic-value-medium-effort)
   1. [NVIDIA CUDA-Q backend](#21-nvidia-cuda-q-target-backend)
   2. [Qibo backend](#22-qibo-backend)
6. [Tier 3 — Cheap wins / partnerships](#tier-3--cheap-wins--partnerships)
   1. [qBraid platform listing](#31-qbraid-platform-listing)
7. [Tier 4 — Niche / academic, do only if time permits](#tier-4--niche--academic-do-only-if-time-permits)
   1. [QuEST adapter](#41-quest-adapter)
   2. [Qulacs adapter](#42-qulacs-adapter)
   3. [Intel-QS adapter](#43-intel-qs-adapter)
   4. [Yao.jl — skip](#44-yaojl--skip)
8. [Recommended sequence and timeline](#recommended-sequence-and-timeline)
9. [Repository organisation](#repository-organisation)
10. [Acceptance criteria per phase](#acceptance-criteria-per-phase)

---

## Strategic principle

**Ship LRET as a family of thin adapter packages**, each implementing the target ecosystem's plugin/extension interface, all depending on a single canonical `qlret` Python package that wraps the C++ core. **Do not attempt to merge LRET source code into upstream core repositories.** Cirq, Aer, and Braket maintainers will reject such PRs; they want third-party plugins instead.

Each upstream PR is therefore a **documentation-only PR** that adds your package to their "Related projects" / "Plugins" / "Compatible simulators" list. The real engineering work lives in your own package repos.

---

## Cross-cutting prerequisites (do these first)

These polish items unblock **every** Tier 1 submission. Skipping them risks reviewer rejection.

| # | Prerequisite | Why it matters | Effort |
|---|---|---|---|
| **P1** | **Fix the silent-death issue in the subprocess backend** | A device that dies on multi-day runs without a traceback fails every reviewer's overnight smoke test. Switch the default path to the in-process pybind11 binding when no `parallel_mode` override is requested; keep subprocess only for explicit mode/threads overrides. | 1–2 days |
| **P2** | **PyPI wheels for Linux + macOS + Windows** | Required for `pip install qlret` to work outside our build environment. Use [cibuildwheel](https://cibuildwheel.readthedocs.io/) + GitHub Actions matrix builds (Ubuntu, macOS-13, macOS-14 arm64, Windows 2022). | 3–5 days |
| **P3** | **Pass PennyLane device test suite** | Gating for the official plugin gallery. Run `pytest pennylane/devices/tests --device=qlret.mixed` and fix any failures. Most should already pass after Phase 1 of the Round 3 plan. | 1–2 days |
| **P4** | **Plumb `parallel_mode=gpu` and `use_gpu=True` through the PennyLane device** | Required for the NVIDIA path; nice-to-have for everyone else. The C++ backend already builds with `USE_GPU=ON`; expose a `--gpu` CLI flag and forward it via `api.simulate_json(parallel_mode='gpu')`. | 2–3 days |
| **P5** | **Round out the test suite** | Currently [python/tests/integration/test_pennylane_device.py](python/tests/integration/test_pennylane_device.py) is light. Add tests for: (a) all 6 parallel modes producing identical states, (b) every supported noise channel, (c) shot-based sampling, (d) gradient via `qml.grad`. | 3–5 days |
| **P6** | **Reproducibility doc + a 30-second demo notebook** | Reviewers want to run a notebook and see the speedup themselves. Write `examples/pennylane_lret_vs_default_mixed.ipynb` with N=10 vqe_noisy and a clear 17× plot. | 2 days |
| **P7** | **Stable v1.0 release notes + semantic versioning policy** | Reviewers won't list a package without a stable version contract. Tag `v1.0.0`. | 1 day |
| **P8** | **Verify license headers + attribution** | Confirm MIT applies to all new code. Citation to Chen, Farquhar, Parrish 2021 (*npj Quantum Information* 7, 61) must appear in README and module docstrings. | 1 day |

**Total prerequisite work: ~2–3 weeks.** None of the Tier 1 targets should be attempted before P1–P3 complete (other prerequisites can run in parallel with Tier 1 work).

---

## Priority ranking — at a glance

| Rank | Target | Tier | Audience reach | Acceptance odds | Effort | Net ROI |
|---|---|---|---|---|---|---|
| 1 | **PennyLane plugin gallery** | 1 | ~50k active | Very high | 2 weeks (mostly already done) | ★★★★★ |
| 2 | **Qiskit BackendV2 adapter** | 1 | ~100k active | High (standalone), low (Aer-core merge) | 4–6 weeks | ★★★★★ |
| 3 | **Cirq simulator adapter** | 1 | ~10k active | High (standalone) | 2–3 weeks | ★★★★ |
| 4 | **Braket local-simulator plugin** | 1 | ~5–10k active | High | 2 weeks | ★★★★ |
| 5 | **NVIDIA CUDA-Q backend** | 2 | ~3–5k emerging | Medium (requires P4) | 6–8 weeks (depends on P4 + GPU work) | ★★★★ if GPU works |
| 6 | **Qibo backend** | 2 | ~1k | Very high (they solicit backends) | 1–2 weeks | ★★★ |
| 7 | **qBraid platform listing** | 3 | ~few k on platform | Highest (just email) | 2 days | ★★★ (very cheap) |
| 8 | **QuEST adapter** | 4 | academic | Medium | 2 weeks | ★★ |
| 9 | **Qulacs adapter** | 4 | ~500 (Japan focus) | Medium | 2 weeks | ★★ |
| 10 | **Intel-QS adapter** | 4 | small | Low | 3 weeks | ★ |
| 11 | **Yao.jl rewrite** | — | Julia niche | n/a | 3+ months (language rewrite) | ✗ skip |

ROI ranking factors: `(reach × acceptance × strategic value) / effort`.

---

## Tier 1 — Highest returns, do first

---

### 1.1 PennyLane plugin gallery (Xanadu)

**Status: ~70% done.** The `qlret.mixed` device is registered as an entry point in [python/setup.py:60-62](python/setup.py) and works end-to-end against the PennyLane Device V2 API.

**What remains:**

1. Complete prerequisites P1–P5 (silent-death fix, PyPI wheels, test-suite pass, optional GPU plumb).
2. Polish device metadata: add `description`, `version`, `pennylane_requires` to the package metadata. Ensure `qml.about()` prints LRET cleanly.
3. Write the **dual-license-friendly docs page** at https://docs.pennylane.ai/projects/qlret/ (Xanadu hosts plugin docs on Read the Docs).
4. Submit pull requests:
   - **PR 1 (their gallery repo):** https://github.com/PennyLaneAI/pennylane.ai-new — add a tile in `/_pennylane_plugins/`. Single file, ~30-line PR.
   - **PR 2 (PennyLane docs):** https://github.com/PennyLaneAI/pennylane — add LRET to the "External plugins" table in `doc/development/plugins.rst`. Single line PR.

**Architectural notes:**

- Device class is in [python/qlret/pennylane_device.py](python/qlret/pennylane_device.py). Already accepts `parallel_mode`, `num_threads` (Phase 2 of Round 3).
- Operation map covers: H, X, Y, Z, S, T, SX, RX, RY, RZ, U1, U2, U3, CNOT, CY, CZ, SWAP, ISWAP, PhaseFlip, BitFlip, AmplitudeDamping, PhaseDamping, GeneralizedAmplitudeDamping, DepolarizingChannel, PauliError, QubitChannel.
- Observables: PauliX/Y/Z, Identity, Hermitian.
- Need to verify: ResetMP, ProbabilityMP, MutualInfoMP measurement support.

**Estimated effort beyond prerequisites:** **3–5 days** for polish, docs, and PR.

**Risk:** Low. Xanadu actively promotes third-party plugins.

**Acceptance criteria:** Listed on https://pennylane.ai/plugins; `pip install qlret[pennylane]` works on all three OSes; PennyLane device test suite passes.

---

### 1.2 Qiskit BackendV2 adapter (IBM)

**Approach: ship a standalone `qiskit-qlret` package, not a PR into Aer.** Aer's core has its own simulator method system (`statevector`, `density_matrix`, `MPS`, etc.) that demands a 3–6 month port. The BackendV2 path is a fraction of that effort and reaches the same audience.

**Implementation steps:**

1. Create a new package repo: `qiskit-qlret/` (sibling to LRET, eventually moved to `github.com/qlret/qiskit-qlret`).
2. Implement `qiskit.providers.BackendV2`:
   ```python
   class QLRETBackend(BackendV2):
       def __init__(self, n_qubits=None, parallel_mode='auto', num_threads=0):
           super().__init__(...)
           self._target = Target(num_qubits=n_qubits, ...)
           # configure basis gates: ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz',
           #                         'cx', 'cz', 'swap', 'depolarizing_channel', ...]

       def run(self, circuits, **run_options) -> Job:
           # translate qiskit Circuit → LRET native ops via _circuit_to_qlret()
           # call qlret.api.simulate_json(...)
           # wrap result in Result object
   ```
3. Implement `QLRETJob(JobV1)` and `Result` wrappers.
4. Implement a `QLRETProvider` so users can `QLRETProvider().get_backend('qlret_mixed')`.
5. Translation utility `qiskit_circuit_to_qlret_ops()` — mostly mirrors what `pennylane_device.py::_op_to_json` already does, but for Qiskit's `Instruction` objects. Handle Qiskit noise channels: `noise.depolarizing_error`, `amplitude_damping_error`, `phase_damping_error`, custom `Kraus` instructions.
6. Compatibility with `qiskit-aer`'s noise-model object: the Aer team has a stable `NoiseModel` API; LRET should accept it and translate.
7. Tests: clone Aer's test suite for new simulator methods and adapt them to QLRETBackend. Aer's own backend tests are at `qiskit_aer/test/terra/backends/`.
8. Package + publish: `pip install qiskit-qlret`. Conda-forge recipe.

**Submission paths (in order of value):**

- **PR A (docs only):** PR to `qiskit-aer` README adding `qiskit-qlret` to the "Related projects" / "Third-party simulators" section. 5-line diff.
- **PR B (docs only):** PR to `qiskit-community/` org docs ([qiskit/SDK ecosystem page](https://qiskit.github.io/ecosystem/)) registering `qiskit-qlret` as an ecosystem member. Single YAML add.
- **PR C (long-term, optional):** Attempt to merge LRET as a true Aer method (`AerSimulator(method='low_rank_density_matrix')`). 3–6 months, low success odds — not on the critical path.

**Estimated effort:** **4–6 weeks** for the adapter package + ecosystem PRs.

**Risk:** Medium for the adapter package (mostly mechanical Qiskit boilerplate); high for the optional Aer-core merge (skip).

**Acceptance criteria:** `pip install qiskit-qlret` works; `QLRETProvider().get_backend('qlret_mixed').run(circuit).result()` matches `qiskit-aer`'s `density_matrix` method to within 1e-10 Frobenius on a 6-qubit noiseless circuit; listed in Qiskit ecosystem registry.

---

### 1.3 Cirq simulator adapter (Google)

**Approach: ship `cirq-qlret` as a standalone package implementing `cirq.SimulatesFinalState`.** Cirq core is intentionally small and conservative about adding simulators.

**Implementation steps:**

1. Create `cirq-qlret/` package repo.
2. Implement the simulator interface:
   ```python
   class QLRETSimulator(
       cirq.SimulatesSamples,
       cirq.SimulatesFinalState,
       cirq.SimulatesIntermediateState,
   ):
       def __init__(self, parallel_mode='auto', num_threads=0): ...

       def _simulate_sweep_iter(self, program, params, qubit_order, initial_state):
           # 1. translate cirq.Circuit → LRET ops
           # 2. call qlret.api.simulate_json(...)
           # 3. yield QLRETTrialResult
           ...
   ```
3. Implement `QLRETTrialResult` with `.final_density_matrix` (an `np.ndarray`) reconstructed from L.
4. Circuit translator: walk `cirq.Operation` objects, map gate types (`cirq.H`, `cirq.X`, `cirq.depolarize`, `cirq.amplitude_damp`, etc.) to LRET's gate names. Reuse the qubit-ordering convention (`cirq.LineQubit.range(n)` → LRET MSB-on-q0) that we already nailed in [benchmarks/_lret_diagnose.py](benchmarks/_lret_diagnose.py).
5. Tests: run Cirq's `cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent` and `assert_same_state_vector` against `cirq.DensityMatrixSimulator` at small N.
6. Publish to PyPI.

**Submission paths:**

- **PR A:** PR to `Cirq` README — add `cirq-qlret` to "Related projects". 3-line diff.
- **PR B:** Submit a talk to the Cirq Bootcamp / QSim Friday seminar series (Google quarterly events) — not a code PR, but high visibility.

**Estimated effort:** **2–3 weeks** for the package + PR.

**Risk:** Low. Cirq's third-party-simulator path is well-trodden (e.g., `qsim-cirq`, `pyquil-cirq`).

**Acceptance criteria:** `pip install cirq-qlret` works; `QLRETSimulator().simulate(circuit).final_density_matrix` matches `cirq.DensityMatrixSimulator` to 1e-10 Frobenius at N=6; listed in Cirq's "Related projects".

---

### 1.4 Braket local-simulator plugin (AWS)

**Approach: ship `amazon-braket-simulator-qlret` implementing the `braket.simulator.BraketSimulator` Python interface.** AWS's reference implementations are open-sourced at https://github.com/amazon-braket/.

**Implementation steps:**

1. Create `amazon-braket-simulator-qlret/` package repo.
2. Implement the BraketSimulator interface (see https://github.com/amazon-braket/amazon-braket-simulator-v1-python for the reference):
   ```python
   from braket.simulator import BraketSimulator
   from braket.task_result import GateModelTaskResult

   class QLRETBraketSimulator(BraketSimulator):
       DEVICE_ID = "braket_local_qlret_mixed"

       def run(self, circuit_ir, shots=0, *, inputs=None, **kwargs) -> GateModelTaskResult:
           # circuit_ir is an OpenQASM-3 string OR a JAQCD program
           # 1. parse to LRET ops (use braket.ir.openqasm.Program parsing)
           # 2. call qlret.api.simulate_json(...)
           # 3. wrap into GateModelTaskResult with measurements / expectation values
           ...
   ```
3. Use AWS's `local_simulator` entry-point registration:
   ```toml
   [project.entry-points."braket.simulators"]
   braket_local_qlret_mixed = "qlret_braket:QLRETBraketSimulator"
   ```
4. Tests: AWS provides a test suite that any local simulator must pass at `amazon-braket-default-simulator-python/test/`. Run that against QLRET-Braket.
5. Publish to PyPI.

**Submission paths:**

- **PR A:** PR to `amazon-braket-sdk-python` README adding `qlret_mixed` to the "Compatible local simulators" list.
- **PR B:** Email AWS Braket partner team for inclusion in the official partner directory.

**Estimated effort:** **2 weeks**.

**Risk:** Low. AWS encourages third-party local simulators.

**Acceptance criteria:** `LocalSimulator("braket_local_qlret_mixed").run(...)` works; AWS's local-simulator test suite passes.

---

## Tier 2 — High strategic value, medium effort

---

### 2.1 NVIDIA CUDA-Q target backend

**Status: requires P4 (GPU plumb-through) to be useful.**

CUDA-Q (formerly cuQuantum Python) is NVIDIA's open-source quantum framework. It accepts custom **target backends** — third-party simulator integrations that CUDA-Q programs can select via `nvq++ --target qlret-gpu`. Source at https://github.com/NVIDIA/cuda-quantum.

**Implementation steps:**

1. Verify the C++ `USE_GPU=ON` build produces a working `quantum_sim --gpu` mode.
2. Add the GPU mode to the Python device (P4 prerequisite).
3. Read NVIDIA's "Adding a custom target" guide in `docs/sphinx/using/extending/`.
4. Build a CUDA-Q `quake-to-qlret` translator (Quake is CUDA-Q's MLIR dialect for quantum circuits).
5. Register as a CUDA-Q target via `runtime/cudaq/platform/...`.
6. Submit as a PR to CUDA-Q `master`.

**Submission path:** Direct PR to NVIDIA's CUDA-Q repo. They actively review third-party target contributions.

**Estimated effort:** **6–8 weeks** total (2–3 weeks GPU plumb + 3–5 weeks CUDA-Q integration).

**Risk:** Medium. The GPU plumb-through requires real C++ work and verification that LRET on GPU is actually competitive with cuStateVec for mixed-state evolution.

**Strategic upside:** Very high. NVIDIA promotes accepted CUDA-Q backends at GTC and on their developer blog. Could also lead to NVIDIA Inception membership and downstream AWS/Azure partnership opportunities through their NVIDIA Quantum Computing Network program.

**Acceptance criteria:** `nvq++ --target qlret-gpu my_program.cpp` compiles and runs; benchmark shows LRET-GPU outperforms `nvidia-mqpu` density-matrix simulator at N≥10 mixed-state circuits.

---

### 2.2 Qibo backend

Qibo (https://github.com/qiboteam/qibo) is an emerging Italian-led open-source quantum framework with an explicitly pluggable backend system (`tensorflow`, `numpy`, `pytorch`, `qibojit`, `qibolab`). They have actively solicited new backends.

**Implementation steps:**

1. Create `qibo-qlret/` package.
2. Implement Qibo's backend interface (subclass `qibo.backends.numpy.NumpyBackend` or `qibo.backends.abstract.Backend`):
   ```python
   from qibo.backends import Backend
   class QLRETBackend(Backend):
       def execute_circuit(self, circuit, initial_state=None, nshots=None): ...
       def execute_density_matrix(self, circuit, initial_state=None): ...
       # implement: matrix, set_seed, apply_gate, apply_channel, ...
   ```
3. Tests: Qibo provides a `pytest tests/test_backends.py` style suite — fork their tests and verify against LRET.
4. Submit:
   - **PR A:** Add `qibo-qlret` to their `qibo/extras/` plugin registry.
   - **PR B:** Submit a paper/preprint with Qibo + LRET noisy benchmarks (they have a publication track record).

**Estimated effort:** **1–2 weeks**.

**Risk:** Very low. Qibo team is welcoming.

**Strategic value:** Medium. Smaller user base than Tier 1, but high acceptance rate and possible co-publication.

**Acceptance criteria:** `qibo.set_backend('qlret')` works; passes Qibo's backend conformance tests.

---

## Tier 3 — Cheap wins / partnerships

---

### 3.1 qBraid platform listing

qBraid (https://qbraid.com) provides a cloud Jupyter environment with one-click access to multiple quantum simulators and hardware. They onboard third-party simulators by **partnership**, not by code PR.

**Steps:**

1. Once `qlret` is on PyPI (after P2), email `partners@qbraid.com` with:
   - Package URL.
   - Benchmark data showing the speedup.
   - Compatibility statement (PennyLane / Qiskit / Cirq adapter availability).
2. They typically respond within a week.
3. If accepted, your package appears in qBraid Lab's simulator dropdown.

**Estimated effort:** **2 days** (mostly waiting on a reply).

**Risk:** Very low.

**Strategic value:** Medium. qBraid is small but is the main platform for non-major-cloud quantum education. Listing there is a "free" credibility boost.

---

## Tier 4 — Niche / academic, do only if time permits

---

### 4.1 QuEST adapter

QuEST (https://github.com/QuEST-Kit/QuEST) is a C/C++ statevector + density-matrix simulator from Oxford. It's used in academic papers but has no plugin system.

**Approach:** Ship `quest-qlret` as a Python wrapper that exposes a QuEST-compatible API (`createDensityQureg`, `applyGate`, etc.) backed by LRET. Users who learned QuEST's API can drop in LRET without rewriting.

**Effort:** 2 weeks. **Acceptance odds:** Low — QuEST doesn't have a plugin slot, but a standalone compatibility package is fine.

**Strategic value:** Low-medium. Useful for the Oxford / academic-physics user base specifically.

---

### 4.2 Qulacs adapter

Qulacs (https://github.com/qulacs/qulacs) is a Japanese research-focused C++ simulator with Python bindings. Similar to QuEST in architecture.

**Approach:** `qulacs-qlret` as a thin compatibility shim.

**Effort:** 2 weeks. **Strategic value:** Low. Niche audience but high acceptance odds in the Japanese quantum-research community.

---

### 4.3 Intel-QS adapter

Intel-QS (https://github.com/intel/intel-qs) focuses on Intel's own statevector method. They have not historically merged third-party simulators.

**Approach:** Standalone Python wrapper that exposes Intel-QS-compatible API.

**Effort:** 3 weeks. **Acceptance odds:** Very low. **Skip unless an Intel-specific use case appears.**

---

### 4.4 Yao.jl — skip

Yao.jl (https://github.com/QuantumBFS/Yao.jl) is the most-loved Julia quantum framework. Adapting LRET would require a full Julia rewrite (Julia cannot trivially call our C++ via `ccall` for the Eigen-heavy parts). **3+ months of work for a small audience. Skip.**

---

## Recommended sequence and timeline

Realistic full-time-equivalent estimates assuming one engineer. Adjust accordingly.

```
Week 1-2     | Prerequisites P1-P8 in parallel
Week 3-4     | Polish PennyLane plugin; submit gallery PR (1.1)
Week 5-7     | Qiskit BackendV2 adapter (1.2)
Week 8-9     | Cirq simulator adapter (1.3)
Week 10-11   | Braket local-simulator plugin (1.4)
Week 12      | qBraid partnership email (3.1) + Qibo backend (2.2)
Week 13-18   | NVIDIA CUDA-Q target — depends on GPU mode actually being fast (2.1)
Week 19-22   | QuEST + Qulacs adapters if time permits (4.1, 4.2)
```

**Milestones:**

- **End of week 4:** Officially listed PennyLane plugin. First public credibility marker.
- **End of week 7:** Qiskit ecosystem entry. Biggest audience reach.
- **End of week 11:** All four Tier 1 adapters shipped. LRET reachable from the four dominant quantum frameworks.
- **End of week 18:** NVIDIA CUDA-Q backend (if GPU is competitive). Eligibility for NVIDIA Inception.

---

## Repository organisation

To keep maintenance sane across N adapter packages, organise them as **sibling repos** under a single GitHub org, not as folders inside the LRET monorepo:

```
github.com/qlret/
├── qlret                              ← canonical C++ + Python package (this repo, renamed)
├── pennylane-qlret                    ← thin entry-point only (could stay in canonical)
├── qiskit-qlret                       ← BackendV2 adapter
├── cirq-qlret                         ← Cirq simulator adapter
├── braket-simulator-qlret             ← BraketSimulator adapter
├── cudaq-qlret                        ← CUDA-Q target
└── qibo-qlret                         ← Qibo backend
```

Each adapter:

- Pins to a specific `qlret>=X.Y` version.
- Has its own CI/CD (passes upstream's plugin tests).
- Has its own PyPI presence.
- Has a short README that links back to the canonical `qlret` for algorithmic and benchmark documentation.

**Do not nest these inside LRET.** That breaks GitHub's PR model, as detailed in the open-source-contribution analysis preceding this document.

---

## Acceptance criteria per phase

### Phase 1 acceptance (PennyLane plugin ready to ship)

- [ ] `pip install qlret[pennylane]` works on Ubuntu/macOS/Windows wheels.
- [ ] PennyLane device test suite passes: `pytest pennylane/devices/tests --device=qlret.mixed -x`.
- [ ] Silent-death issue fixed (P1) — verified by running pub_pennylane_registration for 24+ h without interruption.
- [ ] Demo notebook runs end-to-end in < 60 s and reproduces the headline 17× speedup at N=10 vqe_noisy.
- [ ] Gallery PR opened and merged on `pennylane.ai-new`.

### Phase 2 acceptance (one Tier 1 adapter live)

- [ ] At least one of `qiskit-qlret` / `cirq-qlret` / `braket-simulator-qlret` published to PyPI.
- [ ] Adapter passes upstream's own test suite (Qiskit/Cirq/Braket).
- [ ] Documentation PR opened against upstream README.

### Phase 3 acceptance (full Tier 1 coverage)

- [ ] All four Tier 1 adapters shipped.
- [ ] qBraid listing live.
- [ ] One Tier 2 adapter (Qibo or CUDA-Q) underway.

### Phase 4 acceptance (publication-ready)

- [ ] Paper / preprint draft on the LRET benchmark data (the registration run's full data set, once it finishes) at https://arxiv.org/.
- [ ] LRET cited / used in at least one independent published paper or benchmark suite.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| PyPI wheel build fails on macOS / Windows due to Eigen / OpenMP toolchain quirks | Test cibuildwheel matrix locally before submitting; use `delocate` (macOS) and `delvewheel` (Windows) for dependency bundling. |
| Upstream test suite reveals subtle gate-ordering or noise-channel bugs | Treat first failure as a blocker. The Round 3 small-parallel benchmark only tested LRET-vs-LRET consistency; upstream suites will catch LRET-vs-others discrepancies. |
| Acceptance of doc-only PRs takes months | Acceptable. The packages themselves are useful on PyPI without the upstream README pointer. Upstream PRs are bonus credibility, not gating. |
| LRET-GPU not competitive against cuStateVec / cuTensorNet at the qubit counts NVIDIA cares about | Skip CUDA-Q submission; focus Tier 1 + Tier 2 (Qibo) instead. The CPU story is already compelling. |
| Naming collision: `qlret.mixed` vs `qlret` package vs `qlret-*` adapters | Document the naming hierarchy clearly in the canonical README. Use a uniform `qlret-<framework>` adapter naming convention. |

---

## Citation and attribution

All adapter packages must cite the original LRET paper:

> Chen, Y., Farquhar, C., Parrish, R. M. **Low-rank density-matrix evolution for noisy quantum circuits.** *npj Quantum Information* **7**, 61 (2021). https://doi.org/10.1038/s41534-021-00392-4

Plus, when published, a citation to whatever paper LRET's reimplementation + parallel-mode analysis is published as.

---

## Open questions to resolve before starting

1. **GitHub org name** — `qlret`, `lret`, or something else? Affects all repo URLs.
2. **License** — MIT is fine for all adapters, but verify with the LRET project owner.
3. **Funding** — wheel builds + CUDA-Q GPU testing need CI minutes (GitHub Actions free tier may not be enough for matrix builds; budget ~$10–30/month for Linux + macOS-14 arm64 + GPU runners).
4. **Long-term maintainer** — adapters are low-effort but ongoing. Identify who handles incoming issues per adapter.
5. **NVIDIA Inception application** — apply now (free) so it's already approved by the time the CUDA-Q backend ships.

---

*This plan is a living document. Revise as adapter ROI proves out and as upstream frameworks evolve.*
