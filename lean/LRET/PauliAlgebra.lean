-- LRET Pauli Algebra & QEC Proofs
-- Formal proofs of Pauli multiplication table and commutation relations
--
-- Code references:
--   include/qec_types.h  (Pauli enum, PauliString, commutes_with, pauli_mult)
--   src/qec_types.cpp    (pauli_mult, pauli_mult_phase, commutes_with)

import LRET.Basic
import Mathlib.Algebra.Group.Basic

namespace LRET

open Matrix Complex

-- ============================================================
-- Pauli group elements (mirroring qec_types.h Pauli enum)
-- ============================================================
inductive Pauli : Type
  | I : Pauli  -- Identity
  | X : Pauli  -- Pauli-X
  | Y : Pauli  -- Pauli-Y
  | Z : Pauli  -- Pauli-Z
  deriving DecidableEq, Repr

-- ============================================================
-- Pauli multiplication table (ignoring global phase)
-- Mirrors pauli_mult() in qec_types.cpp
-- ============================================================
def pauliMul : Pauli → Pauli → Pauli
  | Pauli.I, p      => p
  | p,       Pauli.I => p
  | Pauli.X, Pauli.X => Pauli.I
  | Pauli.Y, Pauli.Y => Pauli.I
  | Pauli.Z, Pauli.Z => Pauli.I
  | Pauli.X, Pauli.Y => Pauli.Z   -- XY = iZ (phase tracked separately)
  | Pauli.Y, Pauli.X => Pauli.Z   -- YX = -iZ
  | Pauli.Y, Pauli.Z => Pauli.X
  | Pauli.Z, Pauli.Y => Pauli.X
  | Pauli.Z, Pauli.X => Pauli.Y
  | Pauli.X, Pauli.Z => Pauli.Y

-- ============================================================
-- Theorem 5.1: Pauli self-inverse property: P·P = I
-- (all Paulis are their own inverse up to phase)
-- ============================================================
theorem pauli_self_inverse (p : Pauli) : pauliMul p p = Pauli.I := by
  cases p <;> simp [pauliMul]

-- ============================================================
-- Theorem 5.2: Identity is left and right unit
-- ============================================================
theorem pauli_mul_I_right (p : Pauli) : pauliMul p Pauli.I = p := by
  cases p <;> simp [pauliMul]

theorem pauli_mul_I_left (p : Pauli) : pauliMul Pauli.I p = p := by
  cases p <;> simp [pauliMul]

-- ============================================================
-- PauliString: tensor product of single-qubit Paulis on n qubits
-- Mirrors PauliString class in qec_types.h
-- ============================================================
def PauliString (n : ℕ) := Fin n → Pauli

-- Componentwise multiplication of PauliStrings
def pauliStringMul {n : ℕ} (P Q : PauliString n) : PauliString n :=
  fun i => pauliMul (P i) (Q i)

-- Weight of a PauliString: number of non-identity positions
-- (fixed syntax: filter on Finset, then take card)
def weight {n : ℕ} (P : PauliString n) : ℕ :=
  (Finset.univ.filter (fun i => P i ≠ Pauli.I)).card

def weightFn {n : ℕ} (P : PauliString n) : ℕ :=
  (Finset.univ.filter (fun i => P i ≠ Pauli.I)).card

-- ============================================================
-- Commutation for single-qubit Paulis:
-- X and Y anti-commute: XY = -YX
-- X and Z anti-commute: XZ = -ZX
-- Y and Z anti-commute: YZ = -ZY
-- I commutes with everything
-- ============================================================
def pauliAntiCommutes : Pauli → Pauli → Bool
  | Pauli.X, Pauli.Y => true
  | Pauli.Y, Pauli.X => true
  | Pauli.X, Pauli.Z => true
  | Pauli.Z, Pauli.X => true
  | Pauli.Y, Pauli.Z => true
  | Pauli.Z, Pauli.Y => true
  | _,       _       => false

-- ============================================================
-- Theorem 5.3: Pauli commutation criterion for PauliStrings
-- P and Q commute iff the number of positions where they
-- anti-commute is even.
-- Mirrors: commutes_with() in qec_types.h
-- ============================================================
def anticommuteCount {n : ℕ} (P Q : PauliString n) : ℕ :=
  (Finset.univ.filter (fun i => pauliAntiCommutes (P i) (Q i) = true)).card

def pauliStringCommutes {n : ℕ} (P Q : PauliString n) : Prop :=
  anticommuteCount P Q % 2 = 0

-- ============================================================
-- Theorem 5.4: Pauli I commutes with everything
-- ============================================================
theorem I_commutes_all {n : ℕ} (Q : PauliString n) :
    pauliStringCommutes (fun _ => Pauli.I) Q := by
  simp [pauliStringCommutes, anticommuteCount, pauliAntiCommutes]

-- ============================================================
-- Theorem 5.5: PauliString self-commutation (P commutes with P)
-- Every PauliString commutes with itself (0 anti-commuting positions)
-- ============================================================
theorem pauliString_self_commutes {n : ℕ} (P : PauliString n) :
    pauliStringCommutes P P := by
  simp [pauliStringCommutes, anticommuteCount]
  -- The filter set is empty: no Pauli anti-commutes with itself
  have hempty : (Finset.univ.filter (fun i => pauliAntiCommutes (P i) (P i) = true)) = ∅ := by
    apply Finset.filter_false_of_mem
    intro i _
    simp only [Bool.not_eq_true]
    cases (P i) <;> simp [pauliAntiCommutes]
  simp [hempty]

-- ============================================================
-- Theorem 5.6: PauliString multiplication self-inverse
-- P·P = I (identity PauliString)
-- ============================================================
theorem pauliString_self_inverse {n : ℕ} (P : PauliString n) :
    ∀ i, pauliStringMul P P i = Pauli.I := by
  intro i
  simp [pauliStringMul, pauli_self_inverse]

end LRET
