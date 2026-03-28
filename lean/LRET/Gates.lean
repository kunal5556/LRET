-- LRET Gate Unitarity Proofs
-- Formal proofs that all quantum gates are unitary: U†U = UU† = I
--
-- Code references:
--   include/gates_and_noise.h:19,27  (get_single_qubit_gate, get_two_qubit_gate)
--   src/gates_and_noise.cpp          (explicit matrix definitions)

import LRET.Basic
import Mathlib.LinearAlgebra.UnitaryGroup
import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace LRET

open Matrix Complex Real ComplexConjugate

-- ============================================================
-- Gate matrix definitions (mirroring gates_and_noise.cpp)
-- ============================================================

-- Hadamard gate: H = (1/√2) [[1,1],[1,-1]]
noncomputable def H_gate : Matrix (Fin 2) (Fin 2) ℂ :=
  (1 / Real.sqrt 2 : ℝ) • !![1, 1; 1, -1]

-- Pauli-X gate: X = [[0,1],[1,0]]
def X_gate : Matrix (Fin 2) (Fin 2) ℂ := !![0, 1; 1, 0]

-- Pauli-Y gate: Y = [[0,-i],[i,0]]
def Y_gate : Matrix (Fin 2) (Fin 2) ℂ := !![0, -Complex.I; Complex.I, 0]

-- Pauli-Z gate: Z = [[1,0],[0,-1]]
def Z_gate : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, -1]

-- S gate: S = [[1,0],[0,i]]
def S_gate : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, Complex.I]

-- T gate: T = [[1,0],[0,exp(iπ/4)]]
noncomputable def T_gate : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1, 0; 0, Complex.exp (Complex.I * (π / 4 : ℝ))]

-- RX gate: RX(θ) = [[cos(θ/2), -i·sin(θ/2)]; [-i·sin(θ/2), cos(θ/2)]]
noncomputable def RX_gate (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![(↑(Real.cos (θ/2)) : ℂ), -Complex.I * ↑(Real.sin (θ/2));
     -Complex.I * ↑(Real.sin (θ/2)), ↑(Real.cos (θ/2))]

-- RY gate: RY(θ) = [[cos(θ/2), -sin(θ/2)]; [sin(θ/2), cos(θ/2)]]
noncomputable def RY_gate (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![(↑(Real.cos (θ/2)) : ℂ), -(↑(Real.sin (θ/2)) : ℂ);
     ↑(Real.sin (θ/2)), ↑(Real.cos (θ/2))]

-- RZ gate: RZ(θ) = [[exp(-iθ/2), 0]; [0, exp(iθ/2)]]
noncomputable def RZ_gate (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![Complex.exp (-Complex.I * ↑(θ/2)), 0;
     0, Complex.exp (Complex.I * ↑(θ/2))]

-- CNOT gate (4×4)
def CNOT_gate : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 0, 0;
     0, 1, 0, 0;
     0, 0, 0, 1;
     0, 0, 1, 0]

-- CZ gate (4×4)
def CZ_gate : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 0, 0;
     0, 1, 0, 0;
     0, 0, 1, 0;
     0, 0, 0, -1]

-- SWAP gate (4×4)
def SWAP_gate : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 0, 0;
     0, 0, 1, 0;
     0, 1, 0, 0;
     0, 0, 0, 1]

-- ============================================================
-- Unitarity: a matrix U is unitary iff U†U = I and UU† = I
-- ============================================================
def IsUnitary {n : ℕ} (U : Matrix (Fin n) (Fin n) ℂ) : Prop :=
  U.conjTranspose * U = 1 ∧ U * U.conjTranspose = 1

-- ============================================================
-- Theorem 2.1: Pauli-X is unitary
-- X is real symmetric and self-inverse: X = X†, X² = I
-- ============================================================
theorem pauli_x_unitary : IsUnitary X_gate := by
  constructor <;>
  · ext i j; fin_cases i <;> fin_cases j <;>
    simp [X_gate, Matrix.conjTranspose_apply, Matrix.mul_apply,
          Fin.sum_univ_two, Matrix.one_apply, star_zero, star_one]

-- ============================================================
-- Theorem 2.2: Pauli-Y is unitary
-- Y†Y = [[0,i;-i,0]]·[[0,-i;i,0]] = I (uses I·(-I) = 1)
-- ============================================================
theorem pauli_y_unitary : IsUnitary Y_gate := by
  constructor <;>
  · ext i j; fin_cases i <;> fin_cases j <;>
    simp [Y_gate, Matrix.conjTranspose_apply, Matrix.mul_apply,
          Fin.sum_univ_two, Matrix.one_apply, Complex.star_def,
          Complex.ext_iff, Complex.I_sq, Complex.normSq_apply] <;>
    ring

-- ============================================================
-- Theorem 2.3: Pauli-Z is unitary
-- Z is real diagonal ±1: Z = Z†, Z² = I
-- ============================================================
theorem pauli_z_unitary : IsUnitary Z_gate := by
  constructor <;>
  · ext i j; fin_cases i <;> fin_cases j <;>
    simp [Z_gate, Matrix.conjTranspose_apply, Matrix.mul_apply,
          Fin.sum_univ_two, Matrix.one_apply, star_zero, star_one,
          star_neg, star_one]

-- ============================================================
-- Theorem 2.4: CNOT is unitary (CNOT is self-inverse, real permutation)
-- ============================================================
theorem cnot_unitary : IsUnitary CNOT_gate := by
  constructor <;>
  · ext i j; fin_cases i <;> fin_cases j <;>
    simp [CNOT_gate, Matrix.conjTranspose_apply, Matrix.mul_apply,
          Fin.sum_univ_four, Matrix.one_apply, star_zero, star_one]

-- ============================================================
-- Theorem 2.5: SWAP is unitary (SWAP is self-inverse, real permutation)
-- ============================================================
theorem swap_unitary : IsUnitary SWAP_gate := by
  constructor <;>
  · ext i j; fin_cases i <;> fin_cases j <;>
    simp [SWAP_gate, Matrix.conjTranspose_apply, Matrix.mul_apply,
          Fin.sum_univ_four, Matrix.one_apply, star_zero, star_one]

-- ============================================================
-- Theorem 2.6: CZ is unitary (CZ is self-inverse, real diagonal)
-- ============================================================
theorem cz_unitary : IsUnitary CZ_gate := by
  constructor <;>
  · ext i j; fin_cases i <;> fin_cases j <;>
    simp [CZ_gate, Matrix.conjTranspose_apply, Matrix.mul_apply,
          Fin.sum_univ_four, Matrix.one_apply, star_zero, star_one,
          star_neg, star_one]

-- ============================================================
-- Theorem 2.7: RX(θ) is unitary for all θ : ℝ
-- Uses: cos²(θ/2) + sin²(θ/2) = 1, conj(↑r) = ↑r for r : ℝ
-- ============================================================
set_option maxHeartbeats 800000 in
theorem rx_unitary (θ : ℝ) : IsUnitary (RX_gate θ) := by
  constructor <;>
  · ext i j
    fin_cases i <;> fin_cases j <;>
    -- plain simp evaluates Fin-equality if-then-else from Matrix.one_apply via DecidableEq
    -- Disable ofReal_cos/sin to keep entries in ↑(Real.cos/sin) form;
    -- then ofReal_re/im reduce re/im parts to real arithmetic; and_true strips ∧ True
    -- Remaining goals: cos*cos+sin*sin=1 (linear_combination) and ring goals (ring)
    simp [RX_gate, Matrix.conjTranspose_apply, Matrix.mul_apply, Fin.sum_univ_two,
          Matrix.one_apply, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
          Matrix.head_fin_const, Complex.star_def, map_neg, map_mul,
          conj_ofReal, Complex.conj_I, Complex.ext_iff, Complex.mul_re, Complex.mul_im,
          Complex.add_re, Complex.add_im, Complex.neg_re, Complex.neg_im,
          Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im,
          -Complex.ofReal_cos, -Complex.ofReal_sin, and_true] <;>
    -- After the main simp, two kinds of goals remain:
    -- (diagonal) cos^2+sin^2=1 or sin^2+cos^2=1: closed by simp with trig identities
    -- (off-diagonal) ring goals like -(cos*sin)+sin*cos=0: closed by ring
    -- Use (try simp) so "no progress" on ring goals doesn't abort before ring runs
    (try simp [Real.cos_sq_add_sin_sq, Real.sin_sq_add_cos_sq]) <;>
    ring

-- ============================================================
-- Theorem 2.8: RZ(θ) is unitary for all θ : ℝ
-- Uses: star(exp(iz)) → conj(exp(iz)) → exp(conj(iz)) → exp(-iz),
--       then exp(iz)·exp(-iz) = exp(iz + (-iz)) = exp(0) = 1
-- Strategy: disable exp_conj (LR) and add ←exp_conj (RL) to prevent looping;
--           plain simp expands !![...] matrix entries; hone closes exp(a)·exp(b)=1
-- ============================================================
theorem rz_unitary (θ : ℝ) : IsUnitary (RZ_gate θ) := by
  -- simp normalizes ↑(θ/2:ℝ) → ↑θ/2 (ofReal_div) and -I*x → -(I*x) (ring norm)
  -- Precompute conj/star of exp entries in the normalized form
  have hstar_neg : starRingEnd ℂ (Complex.exp (-(Complex.I * ((↑θ : ℂ) / 2)))) =
                   Complex.exp (Complex.I * ((↑θ : ℂ) / 2)) := by
    show conj (Complex.exp (-(Complex.I * (↑θ / 2)))) = Complex.exp (Complex.I * (↑θ / 2))
    rw [← Complex.exp_conj]; congr 1
    simp only [map_neg, map_mul, map_div₀, Complex.conj_I, conj_ofReal, Complex.conj_ofNat]
    ring
  have hstar_pos : starRingEnd ℂ (Complex.exp (Complex.I * ((↑θ : ℂ) / 2))) =
                   Complex.exp (-(Complex.I * ((↑θ : ℂ) / 2))) := by
    show conj (Complex.exp (Complex.I * (↑θ / 2))) = Complex.exp (-(Complex.I * (↑θ / 2)))
    rw [← Complex.exp_conj]; congr 1
    simp only [map_mul, map_div₀, Complex.conj_I, conj_ofReal, Complex.conj_ofNat]
    ring
  have hm : Complex.exp (-(Complex.I * ((↑θ : ℂ) / 2))) *
            Complex.exp (Complex.I * ((↑θ : ℂ) / 2)) = 1 := by
    rw [← Complex.exp_add, show -(Complex.I * (↑θ/2)) + Complex.I * (↑θ/2) = 0 from by ring,
        Complex.exp_zero]
  have hp : Complex.exp (Complex.I * ((↑θ : ℂ) / 2)) *
            Complex.exp (-(Complex.I * ((↑θ : ℂ) / 2))) = 1 := by
    rw [mul_comm]; exact hm
  constructor <;>
  · ext i j; fin_cases i <;> fin_cases j <;>
    simp [RZ_gate, Matrix.conjTranspose_apply, Matrix.mul_apply, Fin.sum_univ_two,
          Matrix.one_apply, map_zero, mul_zero, zero_mul, add_zero, zero_add,
          hstar_neg, hstar_pos, hm, hp]

-- ============================================================
-- Theorem 2.9: Hadamard is unitary
-- H†H = (1/√2)²·[[2,0],[0,2]] = I  using  (1/√2)² = 1/2
-- ============================================================
set_option maxHeartbeats 800000 in
theorem hadamard_unitary : IsUnitary H_gate := by
  have hsq : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  have h_ne : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  -- (1/√2)·(1/√2) = 1/2, proven once to avoid field_simp in the multi-goal context
  have h12 : (1 / Real.sqrt 2 : ℝ) * (1 / Real.sqrt 2) = 1 / 2 := by
    rw [one_div, one_div, ← mul_inv, hsq]
  have hsq2 : Real.sqrt 2 ^ 2 = 2 := by rw [sq]; exact hsq
  constructor <;>
  · ext i j; fin_cases i <;> fin_cases j <;>
    -- plain simp evaluates Fin-equality if-then-else and handles sqrt arithmetic
    -- and_true strips ∧ True left by ext_iff when im-part closes automatically
    simp [H_gate, Matrix.conjTranspose_apply, Matrix.mul_apply, Fin.sum_univ_two,
          Matrix.one_apply, Matrix.smul_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
          Matrix.head_cons, Matrix.head_fin_const, Complex.star_def, map_smul, smul_eq_mul,
          conj_ofReal, map_neg, map_one, Complex.ext_iff, Complex.mul_re, Complex.mul_im,
          Complex.add_re, Complex.add_im, Complex.ofReal_re, Complex.ofReal_im,
          Complex.neg_re, Complex.neg_im, and_true] <;>
    -- Diagonal goals: √2*√2*(1/2)=1 or √2^2*(1/2)=1
    -- Both closed by nlinarith using hsq (√2*√2=2) and hsq2 (√2^2=2)
    -- Off-diagonal goals (0=0) also closed by nlinarith trivially
    nlinarith [hsq, hsq2, sq_nonneg (Real.sqrt 2)]

-- ============================================================
-- Theorem 2.10: Kronecker product of unitaries is unitary
-- (U⊗ₖV)ᴴ(U⊗ₖV) = (Uᴴ⊗ₖVᴴ)(U⊗ₖV) = (UᴴU)⊗ₖ(VᴴV) = I⊗ₖI = I
-- Mathlib: Matrix.conjTranspose_kronecker, Matrix.mul_kronecker_mul,
--          Matrix.one_kronecker_one
-- Note: returns Matrix (Fin m × Fin k) type, not Matrix (Fin (m*k))
-- ============================================================
theorem kronecker_unitary {m k : ℕ}
    (U : Matrix (Fin m) (Fin m) ℂ) (V : Matrix (Fin k) (Fin k) ℂ)
    (hU : IsUnitary U) (hV : IsUnitary V) :
    let W := Matrix.kroneckerMap (· * ·) U V
    W.conjTranspose * W = 1 ∧ W * W.conjTranspose = 1 := by
  obtain ⟨hU1, hU2⟩ := hU
  obtain ⟨hV1, hV2⟩ := hV
  constructor
  · simp only [Matrix.conjTranspose_kronecker, ← Matrix.mul_kronecker_mul, hU1, hV1,
               Matrix.one_kronecker_one]
  · simp only [Matrix.conjTranspose_kronecker, ← Matrix.mul_kronecker_mul, hU2, hV2,
               Matrix.one_kronecker_one]

end LRET
