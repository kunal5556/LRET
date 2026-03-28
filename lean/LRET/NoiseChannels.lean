-- LRET Noise Channel (CPTP Map) Proofs
-- Formal proofs of Kraus completeness: Σ Kᵢ†Kᵢ = I
-- For: Depolarizing, Amplitude Damping, Phase Damping, Bit Flip, Phase Flip
--
-- Code references:
--   src/gates_and_noise.cpp  (get_noise_kraus_operators, apply_noise_to_L)

import LRET.Basic
import LRET.Gates
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace LRET

open Matrix Complex Real

-- ============================================================
-- Kraus completeness: a channel {Kᵢ} is trace-preserving iff
-- Σᵢ Kᵢ†Kᵢ = I
-- ============================================================
def KrausComplete {n : ℕ} (ks : List (Matrix (Fin n) (Fin n) ℂ)) : Prop :=
  (ks.map (fun K => K.conjTranspose * K)).foldl (· + ·) 0 = 1

-- ============================================================
-- Depolarizing channel (single qubit):
-- K₀ = √(1-p)·I,  K₁ = √(p/3)·X,  K₂ = √(p/3)·Y,  K₃ = √(p/3)·Z
-- ============================================================
noncomputable def depolarizing_kraus (p : ℝ) : List (Matrix (Fin 2) (Fin 2) ℂ) :=
  [ (Real.sqrt (1 - p) : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℂ),
    (Real.sqrt (p / 3) : ℝ) • X_gate,
    (Real.sqrt (p / 3) : ℝ) • Y_gate,
    (Real.sqrt (p / 3) : ℝ) • Z_gate ]

-- ============================================================
-- Theorem 3.1: Depolarizing channel Kraus completeness
-- Σ Kᵢ†Kᵢ = (1-p)I + (p/3)X†X + (p/3)Y†Y + (p/3)Z†Z = I
-- Uses: X†X = Y†Y = Z†Z = I,  (√r)² = r  for r ≥ 0
-- ============================================================
theorem depolarizing_kraus_complete (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
    let ks := depolarizing_kraus p
    (ks.map (fun K => K.conjTranspose * K)).foldl (· + ·) 0 = 1 := by
  -- TODO: Full proof requires unfolding foldl on 4 elements, conjTranspose_smul,
  -- X†X = Y†Y = Z†Z = I (Pauli unitarity), Real.mul_self_sqrt, and ring arithmetic
  -- for (1-p) + p/3 + p/3 + p/3 = 1.  The key steps are:
  --   simp [depolarizing_kraus]
  --   unfold foldl map
  --   simp [Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul]
  --   rw [pauli_x_unitary.1, pauli_y_unitary.1, pauli_z_unitary.1]
  --   simp [Real.mul_self_sqrt hp.1, Real.mul_self_sqrt (by linarith : 0 ≤ p/3)]
  --   ring
  sorry -- TODO: conjTranspose_smul + X†X=I + smul arithmetic; documented above

-- ============================================================
-- Amplitude Damping channel:
-- K₀ = [[1, 0], [0, √(1-γ)]],  K₁ = [[0, √γ], [0, 0]]
-- Models energy relaxation (T₁ decay)
-- ============================================================
noncomputable def amplitude_damping_K0 (γ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1, 0; 0, ↑(Real.sqrt (1 - γ))]

noncomputable def amplitude_damping_K1 (γ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![0, ↑(Real.sqrt γ); 0, 0]

-- ============================================================
-- Theorem 3.2: Amplitude damping Kraus completeness
-- K₀†K₀ + K₁†K₁ = I   when  0 ≤ γ ≤ 1
-- ============================================================
theorem amplitude_damping_kraus_complete (γ : ℝ) (hγ : 0 ≤ γ ∧ γ ≤ 1) :
    (amplitude_damping_K0 γ).conjTranspose * amplitude_damping_K0 γ +
    (amplitude_damping_K1 γ).conjTranspose * amplitude_damping_K1 γ = 1 := by
  -- K₀†K₀ = diag(1, 1-γ),  K₁†K₁ = diag(0, γ),  sum = diag(1,1) = I
  ext i j
  fin_cases i <;> fin_cases j <;>
  simp [amplitude_damping_K0, amplitude_damping_K1, Matrix.conjTranspose_apply,
        Matrix.mul_apply, Matrix.add_apply, Fin.sum_univ_two, Matrix.one_apply,
        Complex.star_def, conj_ofReal, Complex.ext_iff, Complex.mul_re, Complex.mul_im,
        Complex.add_re, Complex.add_im, Complex.ofReal_re, Complex.ofReal_im] <;>
  (try ring) <;>
  nlinarith [Real.mul_self_sqrt hγ.1, Real.mul_self_sqrt (by linarith : 0 ≤ 1 - γ),
             Real.sq_sqrt hγ.1, Real.sq_sqrt (by linarith : 0 ≤ 1 - γ)]

-- ============================================================
-- Phase Damping (Dephasing) channel:
-- K₀ = [[1, 0], [0, √(1-lam)]],  K₁ = [[0, 0], [0, √lam]]
-- Models pure dephasing (T₂ without T₁ component)
-- Note: Lean 4 reserves 'λ' as a keyword; parameter renamed to 'lam'
-- ============================================================
noncomputable def phase_damping_K0 (lam : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1, 0; 0, ↑(Real.sqrt (1 - lam))]

noncomputable def phase_damping_K1 (lam : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![0, 0; 0, ↑(Real.sqrt lam)]

-- ============================================================
-- Theorem 3.3: Phase damping Kraus completeness
-- ============================================================
theorem phase_damping_kraus_complete (lam : ℝ) (hlam : 0 ≤ lam ∧ lam ≤ 1) :
    (phase_damping_K0 lam).conjTranspose * phase_damping_K0 lam +
    (phase_damping_K1 lam).conjTranspose * phase_damping_K1 lam = 1 := by
  -- K₀†K₀ = diag(1, 1-lam),  K₁†K₁ = diag(0, lam),  sum = diag(1,1) = I
  ext i j
  fin_cases i <;> fin_cases j <;>
  simp [phase_damping_K0, phase_damping_K1, Matrix.conjTranspose_apply,
        Matrix.mul_apply, Matrix.add_apply, Fin.sum_univ_two, Matrix.one_apply,
        Complex.star_def, conj_ofReal, Complex.ext_iff, Complex.mul_re, Complex.mul_im,
        Complex.add_re, Complex.add_im, Complex.ofReal_re, Complex.ofReal_im] <;>
  (try ring) <;>
  nlinarith [Real.mul_self_sqrt hlam.1, Real.mul_self_sqrt (by linarith : 0 ≤ 1 - lam),
             Real.sq_sqrt hlam.1, Real.sq_sqrt (by linarith : 0 ≤ 1 - lam)]

-- ============================================================
-- Bit Flip channel:
-- K₀ = √(1-p)·I,  K₁ = √p·X
-- ============================================================
noncomputable def bit_flip_kraus (p : ℝ) : List (Matrix (Fin 2) (Fin 2) ℂ) :=
  [ (Real.sqrt (1 - p) : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℂ),
    (Real.sqrt p : ℝ) • X_gate ]

-- ============================================================
-- Theorem 3.4: Bit flip Kraus completeness
-- (1-p)I + p·X†X = (1-p+p)I = I
-- ============================================================
theorem bit_flip_kraus_complete (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
    let ks := bit_flip_kraus p
    (ks.map (fun K => K.conjTranspose * K)).foldl (· + ·) 0 = 1 := by
  simp only [bit_flip_kraus, List.map, List.foldl]
  -- foldl on 2 elements: 0 + K₀†K₀ + K₁†K₁
  -- = (√(1-p))²·I†I + (√p)²·X†X
  -- = (1-p)·I + p·I  (since X†X = I, I†I = I, (√r)²=r)
  -- = I
  simp only [Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul,
             Matrix.conjTranspose_one, Matrix.one_mul, Matrix.mul_one]
  -- X†X = I from pauli_x_unitary
  have hXX : X_gate.conjTranspose * X_gate = 1 := pauli_x_unitary.1
  rw [hXX]
  -- Now: 0 + (√(1-p) * √(1-p)) • 1 + (√p * √p) • 1 = 1
  -- which simplifies via mul_self_sqrt to (1-p)•1 + p•1 = 1
  ext i j
  simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.zero_apply, Matrix.one_apply,
             smul_eq_mul, zero_add]
  -- After smul, goal involves: √(1-p)*√(1-p)*[i=j] + √p*√p*[i=j] = [i=j]
  -- factor out if-then-else; use Real.mul_self_sqrt
  have h1p : Real.sqrt (1 - p) * Real.sqrt (1 - p) = 1 - p :=
    Real.mul_self_sqrt (by linarith)
  have hsp : Real.sqrt p * Real.sqrt p = p :=
    Real.mul_self_sqrt hp.1
  -- The smul for complex scalars: (r : ℝ) • (x : ℂ) = (↑r * x)
  -- After simp the goal should be arithmetic in ℂ components
  split_ifs with h
  · push_cast
    nlinarith [h1p, hsp]
  · push_cast; ring

-- ============================================================
-- Theorem 3.5: Phase flip Kraus completeness
-- K₀ = √(1-p)·I,  K₁ = √p·Z  →  same calculation as bit flip
-- ============================================================
noncomputable def phase_flip_kraus (p : ℝ) : List (Matrix (Fin 2) (Fin 2) ℂ) :=
  [ (Real.sqrt (1 - p) : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℂ),
    (Real.sqrt p : ℝ) • Z_gate ]

theorem phase_flip_kraus_complete (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
    let ks := phase_flip_kraus p
    (ks.map (fun K => K.conjTranspose * K)).foldl (· + ·) 0 = 1 := by
  simp only [phase_flip_kraus, List.map, List.foldl]
  simp only [Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul,
             Matrix.conjTranspose_one, Matrix.one_mul, Matrix.mul_one]
  -- Z†Z = I from pauli_z_unitary
  have hZZ : Z_gate.conjTranspose * Z_gate = 1 := pauli_z_unitary.1
  rw [hZZ]
  ext i j
  simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.zero_apply, Matrix.one_apply,
             smul_eq_mul, zero_add]
  have h1p : Real.sqrt (1 - p) * Real.sqrt (1 - p) = 1 - p :=
    Real.mul_self_sqrt (by linarith)
  have hsp : Real.sqrt p * Real.sqrt p = p :=
    Real.mul_self_sqrt hp.1
  split_ifs with h
  · push_cast
    nlinarith [h1p, hsp]
  · push_cast; ring

-- ============================================================
-- Theorem 3.6: CPTP maps preserve trace
-- If Σ Kᵢ†Kᵢ = I, then Tr(Σ KᵢρKᵢ†) = Tr(ρ)
-- ============================================================
-- Note: Kraus operators for an n-qubit system are 2^n × 2^n = DMatrix n
theorem kraus_preserves_trace {n : ℕ} (ks : List (DMatrix n))
    (h_complete : (ks.map (fun M : DMatrix n =>
        M.conjTranspose * M)).foldl (· + ·) 0 = 1)
    (ρ : DMatrix n) :
    ((ks.map (fun M : DMatrix n =>
        M * ρ * M.conjTranspose)).foldl (· + ·) 0).trace = ρ.trace := by
  -- TODO: Proof by induction on ks:
  -- Base: foldl [] = 0, Tr(0) = 0, and h_complete gives 0 = I which is a contradiction
  --       for non-empty qubit space; handle by noting both sides are 0.
  -- Step: Tr(Σᵢ MᵢρMᵢ†) = Σᵢ Tr(MᵢρMᵢ†)   (trace linearity)
  --                       = Σᵢ Tr(Mᵢ†Mᵢ·ρ)   (trace cyclicity: Tr(ABC)=Tr(CAB))
  --                       = Tr((Σᵢ Mᵢ†Mᵢ)·ρ)  (trace linearity)
  --                       = Tr(I·ρ)            (h_complete)
  --                       = Tr(ρ)
  sorry -- TODO: List.foldl induction + Matrix.trace_add + Matrix.trace_mul_comm + h_complete

end LRET
