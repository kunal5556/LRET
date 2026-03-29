-- LRET Density Matrix Properties
-- Formal proofs for: Hermitian, PSD, Tr=1, LL† factorization, truncation
--
-- Code references:
--   include/simulator.h:153-154  (validate_density_matrix)
--   src/simulator.cpp            (reconstruct_density_matrix, truncate_L)

import LRET.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace LRET

open Matrix Complex

variable {n : ℕ}

-- ============================================================
-- Theorem 1.1: LL† is Hermitian
-- For any matrix L, (LL†)† = LL†
-- ============================================================
theorem llstar_hermitian {m k : ℕ} (L : CMatrix m k) :
    (L * L.conjTranspose).conjTranspose = L * L.conjTranspose := by
  simp [Matrix.conjTranspose_mul, Matrix.conjTranspose_conjTranspose]

-- ============================================================
-- Theorem 1.2: LL† is positive semidefinite (complex version)
-- For any L, LL† satisfies IsPosSemidefC:
--   Re(v† LL† v) = ‖L†v‖² ≥ 0
-- ============================================================
set_option maxHeartbeats 800000 in
theorem llstar_posSemidef {m k : ℕ} (L : CMatrix m k) :
    IsPosSemidefC (L * L.conjTranspose) := by
  constructor
  · exact llstar_hermitian L
  · intro v
    -- Abbreviation: wₖ = (L†v)ₖ = Σᵢ star(Lᵢₖ)·vᵢ
    -- We show the quadratic form = Σₖ conj(wₖ)·wₖ, then Re ≥ 0
    have hre : ∀ z : ℂ, (star z * z).re = Complex.normSq z := fun z => by
      simp only [Complex.normSq_apply, Complex.star_def, Complex.mul_re,
                 Complex.conj_re, Complex.conj_im]; ring
    have hform : ∑ i : Fin m, ∑ j : Fin m,
                   star (v i) * (L * L.conjTranspose) i j * v j =
                 ∑ col : Fin k,
                   star (∑ row : Fin m, star (L row col) * v row) *
                   (∑ row : Fin m, star (L row col) * v row) := by
      simp only [Matrix.mul_apply, Matrix.conjTranspose_apply]
      -- Both sides equal Σcol Σi Σj star(vi)·L(i,col)·star(L(j,col))·vj;
      -- prove each direction separately to avoid simp_rw interaction
      trans (∑ col : Fin k, ∑ i : Fin m, ∑ j : Fin m,
               star (v i) * L i col * star (L j col) * v j)
      · -- LHS → triple sum: distribute products over the inner Σcol, then reorder
        simp_rw [Finset.mul_sum, Finset.sum_mul]
        conv_lhs => arg 2; ext i; rw [Finset.sum_comm]  -- Σj Σcol → Σcol Σj
        rw [Finset.sum_comm]                             -- Σi Σcol → Σcol Σi
        congr 1; ext col; congr 1; ext i; congr 1; ext j; ring
      · -- RHS → triple sum: expand star-of-sum, then distribute products
        simp_rw [star_sum, StarMul.star_mul, star_star, Finset.sum_mul, Finset.mul_sum]
        congr 1; ext col; congr 1; ext i; congr 1; ext j; ring
    rw [hform, Complex.re_sum]
    exact Finset.sum_nonneg fun col _ => hre _ ▸ Complex.normSq_nonneg _

-- ============================================================
-- Theorem 1.3: If Tr(L†L) = 1, then Tr(LL†) = 1
-- (trace is cyclic: Tr(AB) = Tr(BA))
-- ============================================================
theorem llstar_trace_eq_lstarl_trace {m k : ℕ} (L : CMatrix m k) :
    (L * L.conjTranspose).trace = (L.conjTranspose * L).trace := by
  rw [Matrix.trace_mul_comm]

-- ============================================================
-- Theorem 1.4: Gram matrix G = L†L is positive semidefinite
-- This is the Gram matrix used in LRET's rank truncation
-- ============================================================
set_option maxHeartbeats 800000 in
theorem gram_matrix_posSemidef {m k : ℕ} (L : CMatrix m k) :
    IsPosSemidefC (L.conjTranspose * L) := by
  constructor
  · show (L.conjTranspose * L).conjTranspose = L.conjTranspose * L
    rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_conjTranspose]
  · intro v
    -- wᵣ = (Lv)ᵣ = Σcol L r col * v col; quadratic form = Σᵣ conj(wᵣ)·wᵣ
    have hre : ∀ z : ℂ, (star z * z).re = Complex.normSq z := fun z => by
      simp only [Complex.normSq_apply, Complex.star_def, Complex.mul_re,
                 Complex.conj_re, Complex.conj_im]; ring
    have hform : ∑ i : Fin k, ∑ j : Fin k,
                   star (v i) * (L.conjTranspose * L) i j * v j =
                 ∑ row : Fin m,
                   star (∑ col : Fin k, L row col * v col) *
                   (∑ col : Fin k, L row col * v col) := by
      simp only [Matrix.mul_apply, Matrix.conjTranspose_apply]
      trans (∑ row : Fin m, ∑ i : Fin k, ∑ j : Fin k,
               star (v i) * star (L row i) * L row j * v j)
      · simp_rw [Finset.mul_sum, Finset.sum_mul]
        conv_lhs => arg 2; ext i; rw [Finset.sum_comm]  -- Σj Σrow → Σrow Σj
        rw [Finset.sum_comm]                             -- Σi Σrow → Σrow Σi
        congr 1; ext row; congr 1; ext i; congr 1; ext j; ring
      · simp_rw [star_sum, StarMul.star_mul, Finset.sum_mul, Finset.mul_sum]
        congr 1; ext row; congr 1; ext i; congr 1; ext j; ring
    rw [hform, Complex.re_sum]
    exact Finset.sum_nonneg fun row _ => hre _ ▸ Complex.normSq_nonneg _

-- ============================================================
-- Theorem 1.5: Density matrix validity from LL† factorization
-- If Tr(L†L) = 1, then ρ = LL† is a valid density matrix
-- L is constrained to Fin (2^n) rows so that LL† : DMatrix n
-- ============================================================
theorem lret_density_matrix_valid {n k : ℕ} (L : CMatrix (2^n) k)
    (h_trace : (L.conjTranspose * L).trace = 1) :
    IsDensityMatrix (L * L.conjTranspose) where
  hermitian  := llstar_hermitian L
  posSemiDef := llstar_posSemidef L
  unit_trace := by rw [Matrix.trace_mul_comm]; exact h_trace

-- ============================================================
-- Theorem 1.6: Measurement probability is non-negative
-- Re(v† ρ v) ≥ 0 for any state v and density matrix ρ
-- (directly from IsDensityMatrix's IsPosSemidefC component)
-- ============================================================
theorem measurement_prob_nonneg {n : ℕ} (ρ : DMatrix n)
    (hρ : IsDensityMatrix ρ) (v : Fin (2^n) → ℂ) :
    0 ≤ (∑ i : Fin (2^n), ∑ j : Fin (2^n), star (v i) * ρ i j * v j).re :=
  hρ.posSemiDef.2 v

-- ============================================================
-- Theorem 1.7: Purity upper bound
-- Tr(ρ²) ≤ 1 for any valid density matrix
-- (follows from PSD + trace-1: eigenvalues ≥ 0, sum = 1 → sum of squares ≤ 1)
-- ============================================================
theorem purity_le_one {n : ℕ} (ρ : DMatrix n) (hρ : IsDensityMatrix ρ) :
    ((ρ * ρ).trace).re ≤ 1 := by
  -- Purity Tr(ρ²) = Σ λᵢ² where λᵢ are eigenvalues
  -- Since λᵢ ≥ 0 and Σ λᵢ = 1, by Cauchy-Schwarz: Σ λᵢ² ≤ (Σ λᵢ)² = 1
  -- Requires spectral decomposition from Mathlib; leave as documented sorry
  sorry -- Requires spectral theorem; leave as documented sorry

end LRET
