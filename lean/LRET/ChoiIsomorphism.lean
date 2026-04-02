-- LRET Choi Isomorphism Theorem
-- Proves: (U⊗conj(U)) · vec(ρ) = vec(UρU†)
-- This formally verifies the core of LRET's gate evolution mechanism.
--
-- Code reference:
--   src/simulator.cpp  (apply_single_qubit_gate, apply_two_qubit_gate)

import LRET.Basic
import LRET.Gates
import Mathlib.LinearAlgebra.Matrix.Kronecker
import Mathlib.LinearAlgebra.Matrix.ConjTranspose

namespace LRET

open Matrix Complex

variable {n : ℕ}

-- ============================================================
-- Row-major vectorization: vec(ρ) maps a 2^n × 2^n matrix
-- to a (Fin 2^n × Fin 2^n)-indexed function by pairing indices.
-- Index mapping: vec(ρ) (i, j) = ρ i j
-- (Using product indices avoids Fin(4^n) arithmetic.)
-- ============================================================
noncomputable def vecDensity {n : ℕ} (ρ : DMatrix n) : (Fin (2^n) × Fin (2^n)) → ℂ :=
  fun ⟨i, j⟩ => ρ i j

-- ============================================================
-- The Choi matrix for a unitary U: C_U = U ⊗ conj(U)
-- where conj(U) = star applied elementwise (= U̅, complex conjugate)
-- Returns type: Matrix (Fin 2^n × Fin 2^n) (Fin 2^n × Fin 2^n) ℂ
-- ============================================================
noncomputable def choiMatrix {n : ℕ} (U : DMatrix n) :
    Matrix (Fin (2^n) × Fin (2^n)) (Fin (2^n) × Fin (2^n)) ℂ :=
  Matrix.kroneckerMap (· * ·) U (U.map star)

-- ============================================================
-- Theorem 5.1: Choi isomorphism for gate evolution
-- (U ⊗ conj(U)) · vec(ρ) = vec(U · ρ · U†)
-- This is the mathematical foundation for LRET's gate application:
-- instead of forming the full density matrix, LRET evolves L directly.
--
-- Proof sketch:
--   [(U⊗conj(U)) · vec(ρ)]_{ij} = Σ_kl (U⊗conj(U))_{ij,kl} · ρ_{kl}
--                                 = Σ_kl U_{ik} · conj(U_{jl}) · ρ_{kl}
--                                 = (UρU†)_{ij}
--                                 = [vec(UρU†)]_{ij}
-- Full proof: simp to double sums, then sum_comm + ring.
-- TODO: use Fintype.sum_prod_type, Finset.sum_mul, Finset.sum_comm, ring.
-- ============================================================
set_option maxHeartbeats 400000 in
theorem choi_gate_evolution {n : ℕ} (U : DMatrix n) (ρ : DMatrix n) :
    (choiMatrix U).mulVec (vecDensity ρ) = vecDensity (U * ρ * U.conjTranspose) := by
  funext ⟨i, j⟩
  -- Unfold definitions to double sums
  -- Matrix.mulVec unfolds to ⬝ᵥ (dotProduct); add dotProduct to expand to Finset.sum
  simp only [choiMatrix, vecDensity, Matrix.mulVec, dotProduct,
             Matrix.kroneckerMap_apply, Matrix.map_apply, Matrix.mul_apply,
             Matrix.conjTranspose_apply]
  -- LHS: Σ_{x : Fin(2^n) × Fin(2^n)}, U i x.1 * star(U j x.2) * ρ x.1 x.2
  -- Convert product-type sum to double sum
  rw [Fintype.sum_prod_type]
  -- LHS: Σ_a Σ_b, U i a * star(U j b) * ρ a b
  -- RHS: Σ_k (Σ_l U i l * ρ l k) * star(U j k)
  -- Distribute * star(U j k) inside inner sum on RHS
  simp_rw [Finset.sum_mul]
  -- RHS: Σ_k Σ_l, U i l * ρ l k * star(U j k)
  -- Swap the two RHS sums to align indices with LHS
  conv_rhs => rw [Finset.sum_comm]
  -- RHS: Σ_l Σ_k, U i l * ρ l k * star(U j k)   (l = a, k = b after rename)
  -- Both sides now have form Σ_a Σ_b, <terms equal by ring>
  congr 1; ext a; congr 1; ext b; ring

-- ============================================================
-- Corollary 5.2: Gate evolution preserves density matrix properties
-- If ρ is a valid density matrix and U is unitary, UρU† is also valid.
-- ============================================================
theorem gate_preserves_density_matrix {n : ℕ} (U : DMatrix n) (ρ : DMatrix n)
    (hU : IsUnitary U) (hρ : IsDensityMatrix ρ) :
    IsDensityMatrix (U * ρ * U.conjTranspose) where
  hermitian := by
    -- Goal: (U * ρ * U†)ᴴ = U * ρ * U†
    -- (U * ρ * U†)ᴴ = (U†)ᴴ * (U * ρ)ᴴ = U * (ρᴴ * U†) = U * ρ * U†
    rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_mul,
        Matrix.conjTranspose_conjTranspose, hρ.hermitian, ← Matrix.mul_assoc]
  posSemiDef := by
    constructor
    · -- Goal: (U * ρ * U†).IsHermitian   (= (U * ρ * U†)ᴴ = U * ρ * U†)
      -- IsHermitian is a def, so unfold it to expose the ᴴ = form
      show (U * ρ * U.conjTranspose).conjTranspose = U * ρ * U.conjTranspose
      rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_mul,
          Matrix.conjTranspose_conjTranspose, hρ.hermitian, ← Matrix.mul_assoc]
    · intro v
      -- Re(v†(UρU†)v) = Re((U†v)†ρ(U†v)) ≥ 0  by PSD of ρ applied to w = U†v
      have hpsd := hρ.posSemiDef.2 (U.conjTranspose.mulVec v)
      -- suffices: show goal sum equals hypothesis sum, then use hpsd
      suffices heq : ∑ i : Fin (2^n), ∑ j : Fin (2^n),
            star (v i) * (U * ρ * U.conjTranspose) i j * v j =
          ∑ i : Fin (2^n), ∑ j : Fin (2^n),
            star ((U.conjTranspose.mulVec v) i) * ρ i j *
              (U.conjTranspose.mulVec v) j by
        rw [heq]; exact hpsd
      -- Expand U†v and UρU† entrywise; expand star(Σ ...) on RHS
      simp only [Matrix.mulVec, dotProduct, Matrix.conjTranspose_apply, Matrix.mul_apply]
      simp_rw [star_sum, StarMul.star_mul, star_star, Finset.sum_mul, Finset.mul_sum]
      -- Both sides are 4-fold sums ∑_i ∑_j ∑_k ∑_l with the same terms under index renaming:
      -- LHS: ∑_i ∑_j ∑_k ∑_l, star(v_i) * U_il * ρ_lk * star(U_jk) * v_j
      -- RHS: ∑_i ∑_j ∑_k ∑_l, star(v_k) * U_ki * ρ_ij * star(U_lj) * v_l
      -- Rename LHS indices (i→k, j→l, l→i, k→j) to match RHS: requires swapping ∑∑ pairs.
      -- TODO: close by Finset.sum_comm (swap outer pair past inner pair) + ring
      sorry -- TODO: 4-fold sum reindex: LHS(i,j,k,l)→RHS via (i↔k,j↔l) rename
  unit_trace := by
    rw [Matrix.trace_mul_comm, ← Matrix.mul_assoc, hU.1, Matrix.one_mul]
    exact hρ.unit_trace

end LRET
