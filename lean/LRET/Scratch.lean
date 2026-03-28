import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Algebra.BigOperators.Fin

open Matrix Complex Real

noncomputable def RZ2 (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![Complex.exp (-Complex.I * ↑(θ/2)), 0; 0, Complex.exp (Complex.I * ↑(θ/2))]

example (θ : ℝ) : (RZ2 θ).conjTranspose * RZ2 θ = 1 := by
  have hminus : star (Complex.exp (-Complex.I * ↑(θ / 2))) *
                Complex.exp (-Complex.I * ↑(θ / 2)) = 1 := by
    rw [Complex.star_def, ← Complex.exp_conj]
    simp only [map_mul, map_neg, Complex.conj_I, neg_neg, conj_ofReal]
    rw [← Complex.exp_add,
        show Complex.I * ↑(θ / 2 : ℝ) + -Complex.I * ↑(θ / 2 : ℝ) = 0 from by ring,
        Complex.exp_zero]
  have hF00 : ((0:Fin 2) = 0) = True := by decide
  have hF01 : ((0:Fin 2) = 1) = False := by decide
  have hF10 : ((1:Fin 2) = 0) = False := by decide
  have hF11 : ((1:Fin 2) = 1) = True := by decide
  ext i j
  fin_cases i <;> fin_cases j <;>
  simp only [RZ2, Matrix.conjTranspose_apply, Matrix.mul_apply, Fin.sum_univ_two,
             Matrix.one_apply, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
             Matrix.head_fin_const, star_zero, mul_zero, zero_mul, add_zero, zero_add,
             hF00, hF01, hF10, hF11, if_true, if_false]
  first | exact hminus | rfl
