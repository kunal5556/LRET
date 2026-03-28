-- LRET Basic Types and Quantum Foundations
-- Shared definitions used across all verification modules

import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Complex.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace LRET

open Matrix Complex

variable {n : ℕ}

-- The Hilbert space dimension for n qubits
abbrev HilbertDim (n : ℕ) : ℕ := 2 ^ n

-- Type alias: density matrix is a square complex matrix over 2^n dimensions
abbrev DMatrix (n : ℕ) := Matrix (Fin (2^n)) (Fin (2^n)) ℂ

-- Type alias: n×m complex matrix (for low-rank factor L)
abbrev CMatrix (m k : ℕ) := Matrix (Fin m) (Fin k) ℂ

-- Complex PSD predicate: M is positive semidefinite over ℂ iff
--   (a) M is Hermitian:  Mᴴ = M
--   (b) ∀ v : Fin m → ℂ,  Re(v† M v) ≥ 0
-- Note: Matrix.PosSemidef requires PartialOrder α which ℂ does not have,
--       so we give an equivalent self-contained definition.
def IsPosSemidefC {m : ℕ} (M : Matrix (Fin m) (Fin m) ℂ) : Prop :=
  M.IsHermitian ∧
  ∀ v : Fin m → ℂ,
    0 ≤ (∑ i : Fin m, ∑ j : Fin m, star (v i) * M i j * v j).re

-- Predicate: a matrix is a valid density matrix
-- ρ must be: Hermitian, positive semidefinite, trace = 1
structure IsDensityMatrix (ρ : DMatrix n) : Prop where
  hermitian  : ρ.conjTranspose = ρ
  posSemiDef : IsPosSemidefC ρ
  unit_trace : ρ.trace = 1

-- Standard basis vector |i⟩ as a column vector
def basisVec (i : Fin (2^n)) : Fin (2^n) → ℂ :=
  fun j => if i = j then 1 else 0

-- Projector onto |i⟩: Π_i = |i⟩⟨i|
-- (basisVec i k)* is the complex conjugate = star (basisVec i k)
def projector (i : Fin (2^n)) : DMatrix n :=
  Matrix.of (fun j k => basisVec i j * star (basisVec i k))

-- Trace of a matrix (concrete definition for computation)
noncomputable def mTrace {m : ℕ} (A : Matrix (Fin m) (Fin m) ℂ) : ℂ :=
  Finset.univ.sum (fun i => A i i)

-- Frobenius norm squared: ‖A‖²_F = Σᵢⱼ |Aᵢⱼ|²
-- Complex.normSq : ℂ → ℝ,  so the sum is already ℝ
noncomputable def frobeniusNormSq {m k : ℕ} (A : CMatrix m k) : ℝ :=
  Finset.univ.sum (fun i => Finset.univ.sum (fun j => Complex.normSq (A i j)))

end LRET
