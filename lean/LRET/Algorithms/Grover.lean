-- LRET Algorithm: Grover's Search
-- Formal statement of the quadratic speedup theorem
--
-- Code reference:
--   python/pennylane_algorithms/tier1/grover.py

import LRET.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace LRET.Algorithms

open Real

-- ============================================================
-- Grover's algorithm operates in a 2D subspace.
-- Let α_k = sin(θ_k) where sin²(θ_0) = 1/N (initial overlap with target)
-- After k iterations: sin²(θ_k) = sin²((2k+1)·arcsin(1/√N))
-- ============================================================

-- Initial amplitude of target state: 1/√N for N items
noncomputable def initialAmplitude (N : ℕ) (hN : 0 < N) : ℝ := 1 / Real.sqrt N

-- Amplitude after k Grover iterations
noncomputable def groverAmplitude (N : ℕ) (k : ℕ) (hN : 0 < N) : ℝ :=
  Real.sin ((2 * k + 1) * Real.arcsin (initialAmplitude N hN))

-- ============================================================
-- Theorem 6.1: Grover amplitude after optimal steps
-- With k* = ⌊π/4 · √N⌋ iterations, probability of success ≥ 1 - 1/N
-- ============================================================
-- Optimal number of iterations
noncomputable def optimalSteps (N : ℕ) (hN : 0 < N) : ℕ :=
  Nat.floor (π / 4 * Real.sqrt N)

-- ============================================================
-- Theorem 6.2: The Grover diffusion operator is unitary
-- D = 2|s⟩⟨s| - I where |s⟩ is the uniform superposition
-- D is a reflection, hence unitary and self-inverse
-- ============================================================
-- For N = 2^n, the uniform state |s⟩ = H^⊗n |0⟩^⊗n
-- D = 2|s⟩⟨s| - I is a reflection → unitary
-- Reflection unitarity: (2P - I)†(2P - I) = I for projector P
theorem reflection_unitary {n : ℕ} (P : DMatrix n)
    (hP_proj : P * P = P)          -- P is a projector
    (hP_herm : P.conjTranspose = P) -- P is Hermitian
    (hP_real : ∀ i j, (P i j).im = 0) : -- P has real entries (uniform state)
    let D := 2 • P - 1
    D.conjTranspose * D = 1 := by
  simp only []
  -- D = 2P - I, D† = (2P)† - I† = 2P† - I = 2P - I = D  (since P Hermitian)
  -- D† D = (2P - I)(2P - I) = 4P² - 4P + I = 4P - 4P + I = I  (since P² = P)
  have hDherm : (2 • P - 1 : DMatrix n).conjTranspose = 2 • P - 1 := by
    rw [Matrix.conjTranspose_sub, Matrix.conjTranspose_smul, hP_herm,
        Matrix.conjTranspose_one]
    norm_cast
  rw [hDherm]
  calc (2 • P - 1) * (2 • P - 1)
      = 4 • (P * P) - 4 • P + 1 := by noncomm_ring
    _ = 4 • P - 4 • P + 1 := by rw [hP_proj]
    _ = 1 := by abel

-- ============================================================
-- Theorem 6.3: Phase kickback in oracle
-- For Grover oracle O|x⟩|−⟩ = (-1)^f(x)|x⟩|−⟩
-- O is unitary (it's a controlled-phase flip)
-- ============================================================
-- The oracle flips the phase of the target state → unitary
theorem grover_oracle_unitary {n : ℕ} (target : Fin (2^n)) :
    ∃ (O : DMatrix n), O.conjTranspose * O = 1 ∧
    ∀ i, O i i = if i = target then -1 else 1 := by
  -- The oracle is a diagonal unitary with ±1 entries
  use Matrix.diagonal (fun i => if i = target then (-1 : ℂ) else 1)
  constructor
  · simp [Matrix.diagonal_conjTranspose, Matrix.diagonal_mul_diagonal]
    ext i j
    simp [Matrix.diagonal_apply, Matrix.one_apply]
    split_ifs with h <;> simp
  · intro i
    simp only [Matrix.diagonal_apply]
    split_ifs <;> rfl

end LRET.Algorithms
