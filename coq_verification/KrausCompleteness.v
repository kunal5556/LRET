(* LRET Kraus Completeness — Coq + QuantumLib
   Independent verification of Kraus operator completeness.
   Uses QuantumLib: https://github.com/inQWIRE/QuantumLib

   Install: opam install coq-quantumlib
   Build:   make KrausCompleteness.vo
*)

Require Import QuantumLib.Quantum.
Require Import QuantumLib.Matrix.
Require Import QuantumLib.Complex.

(* ============================================================
   Helper: Scalar multiplication scales Kraus completeness.
   If K†K = I, then (c·K)†(c·K) = |c|² I
   ============================================================ *)
Lemma scale_adjoint_mul : forall (n : nat) (c : C) (K : Matrix n n),
  (c .* K) † × (c .* K) = (c^* * c) .* (K† × K).
Proof.
  intros n c K.
  rewrite Mscale_adj, Mscale_mult_dist_l, Mscale_mult_dist_r, Mscale_assoc.
  reflexivity.
Qed.

(* ============================================================
   Depolarizing channel Kraus operators:
     K0 = √(1-p) · I₂
     K1 = √(p/3) · X
     K2 = √(p/3) · Y
     K3 = √(p/3) · Z

   Theorem: K0†K0 + K1†K1 + K2†K2 + K3†K3 = I₂
   Proof:   (1-p)I + (p/3)X†X + (p/3)Y†Y + (p/3)Z†Z
          = (1-p)I + (p/3)I + (p/3)I + (p/3)I
          = (1-p + p)I = I
   ============================================================ *)
Lemma depolarizing_kraus_complete : forall (p : R),
  0 <= p -> p <= 1 ->
  let K0 := (RtoC (sqrt (1 - p))) .* I 2 in
  let K1 := (RtoC (sqrt (p / 3))) .* σx in
  let K2 := (RtoC (sqrt (p / 3))) .* σy in
  let K3 := (RtoC (sqrt (p / 3))) .* σz in
  K0† × K0 .+ K1† × K1 .+ K2† × K2 .+ K3† × K3 = I 2.
Proof.
  intros p Hp0 Hp1.
  (* Use: X†X = Y†Y = Z†Z = I (from QuantumLib), I†I = I *)
  (* (√r)^* * √r = r for r ≥ 0 *)
  unfold Mmult, Mplus, Mscale, adjoint, I, σx, σy, σz.
  prep_matrix_equality.
  destruct x, y.
  all: simpl.
  all: try (field_simplify; try lra).
  all: repeat rewrite sqrt_sqrt; try lra.
  all: try lra.
Qed.

(* ============================================================
   Amplitude Damping channel:
     K0 = [[1, 0], [0, √(1-γ)]]
     K1 = [[0, √γ], [0, 0]]

   Theorem: K0†K0 + K1†K1 = I₂
   Proof:   K0†K0 = diag(1, 1-γ), K1†K1 = diag(0, γ)
            Sum = diag(1, 1) = I
   ============================================================ *)
Lemma amplitude_damping_kraus_complete : forall (gamma : R),
  0 <= gamma -> gamma <= 1 ->
  let K0 : Matrix 2 2 := fun i j =>
    match i, j with
    | 0, 0 => C1 | 1, 1 => RtoC (sqrt (1 - gamma))
    | _, _ => C0 end in
  let K1 : Matrix 2 2 := fun i j =>
    match i, j with
    | 0, 1 => RtoC (sqrt gamma)
    | _, _ => C0 end in
  K0† × K0 .+ K1† × K1 = I 2.
Proof.
  intros gamma Hg0 Hg1.
  unfold Mmult, Mplus, adjoint, I.
  prep_matrix_equality.
  destruct x, y; simpl; try lca.
  (* (0,0): 1*1 + 0 = 1 *)
  (* (1,1): (√(1-γ))² + (√γ)² = (1-γ) + γ = 1 *)
  all: rewrite ?sqrt_sqrt; try lra.
  all: try lca.
  (* (1,1) case: need (1-γ) + γ = 1 *)
  unfold RtoC; simpl.
  constructor; try lra.
  rewrite sqrt_sqrt; lra.
Qed.

(* ============================================================
   Phase Damping channel:
     K0 = [[1, 0], [0, √(1-λ)]]
     K1 = [[0, 0], [0, √λ]]

   Theorem: K0†K0 + K1†K1 = I₂ (same structure as amplitude damping)
   ============================================================ *)
Lemma phase_damping_kraus_complete : forall (lam : R),
  0 <= lam -> lam <= 1 ->
  let K0 : Matrix 2 2 := fun i j =>
    match i, j with
    | 0, 0 => C1 | 1, 1 => RtoC (sqrt (1 - lam))
    | _, _ => C0 end in
  let K1 : Matrix 2 2 := fun i j =>
    match i, j with
    | 1, 1 => RtoC (sqrt lam)
    | _, _ => C0 end in
  K0† × K0 .+ K1† × K1 = I 2.
Proof.
  intros lam Hl0 Hl1.
  unfold Mmult, Mplus, adjoint, I.
  prep_matrix_equality.
  destruct x, y; simpl; try lca.
  unfold RtoC; simpl.
  constructor; try lra.
  rewrite sqrt_sqrt; lra.
Qed.

(* ============================================================
   Bit Flip channel:
     K0 = √(1-p) · I₂
     K1 = √p · X

   Theorem: K0†K0 + K1†K1 = (1-p)I + p·X†X = (1-p+p)I = I
   ============================================================ *)
Lemma bit_flip_kraus_complete : forall (p : R),
  0 <= p -> p <= 1 ->
  let K0 := (RtoC (sqrt (1 - p))) .* I 2 in
  let K1 := (RtoC (sqrt p)) .* σx in
  K0† × K0 .+ K1† × K1 = I 2.
Proof.
  intros p Hp0 Hp1.
  unfold Mmult, Mplus, Mscale, adjoint, I, σx.
  prep_matrix_equality.
  destruct x, y; simpl; try lca.
  all: unfold RtoC; simpl.
  all: constructor; try lra.
  all: repeat rewrite sqrt_sqrt; try lra.
Qed.

(* ============================================================
   Phase Flip channel:
     K0 = √(1-p) · I₂
     K1 = √p · Z

   Same completeness argument as bit flip (Z†Z = I)
   ============================================================ *)
Lemma phase_flip_kraus_complete : forall (p : R),
  0 <= p -> p <= 1 ->
  let K0 := (RtoC (sqrt (1 - p))) .* I 2 in
  let K1 := (RtoC (sqrt p)) .* σz in
  K0† × K0 .+ K1† × K1 = I 2.
Proof.
  intros p Hp0 Hp1.
  unfold Mmult, Mplus, Mscale, adjoint, I, σz.
  prep_matrix_equality.
  destruct x, y; simpl; try lca.
  all: unfold RtoC; simpl.
  all: constructor; try lra.
  all: repeat rewrite sqrt_sqrt; try lra.
Qed.
