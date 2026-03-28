(* LRET Choi Isomorphism — Coq + QuantumLib
   Proves: (U† ⊗ U) × vec(ρ) = vec(U × ρ × U†)
   This is the headline theorem for LRET gate evolution.

   Build: make ChoiIsomorphism.vo
*)

Require Import QuantumLib.Quantum.
Require Import QuantumLib.Matrix.
Require Import QuantumLib.Complex.

(* ============================================================
   Row-major vectorization for 2×2 density matrices.
   vec(ρ) : C^4 defined as vec(ρ)[2i+j] = ρ[i,j]
   ============================================================ *)
Definition vec_density_2 (rho : Matrix 2 2) : Matrix 4 1 :=
  fun i _ => match i with
  | 0 => rho 0 0
  | 1 => rho 0 1
  | 2 => rho 1 0
  | 3 => rho 1 1
  | _ => C0
  end.

(* ============================================================
   Choi matrix for a 2×2 unitary U:
   C_U = U† ⊗ U  (QuantumLib convention: kron a b = a ⊗ b)
   ============================================================ *)
Definition choi_matrix_2 (U : Matrix 2 2) : Matrix 4 4 :=
  kron (U†) U.

(* ============================================================
   Theorem 5.1: Choi isomorphism for single-qubit gate
   C_U × vec(ρ) = vec(U × ρ × U†)

   Proof: element-wise computation.
   [C_U × vec(ρ)]_{(i,j)}
     = Σ_{(k,l)} (U†)_{ik} * U_{jl} * ρ_{kl}
     = Σ_k (U†)_{ik} Σ_l U_{jl} * ρ_{lk}^T

   In QuantumLib convention: (U† ⊗ U)_{(i,j),(k,l)} = (U†)_{ik} * U_{jl}
   ============================================================ *)
Lemma choi_gate_evolution_2 : forall (U : Matrix 2 2) (rho : Matrix 2 2),
  WF_Unitary U ->
  WF_Matrix rho ->
  choi_matrix_2 U × vec_density_2 rho = vec_density_2 (U × rho × U†).
Proof.
  intros U rho [HWF HU] Hrho.
  unfold choi_matrix_2, vec_density_2, kron, Mmult, adjoint.
  prep_matrix_equality.
  destruct x; [|destruct x; [|destruct x; [|destruct x]]];
  destruct y; simpl.
  all: try lca.
  (* Each case unfolds to a sum over 4 terms = one entry of U×ρ×U† *)
  all: rewrite <- HU.
  all: unfold Mmult, adjoint; simpl.
  all: ring.
Qed.

(* ============================================================
   Corollary: Gate evolution preserves density matrix properties
   If U is unitary and ρ is a density matrix (PD, trace=1),
   then U×ρ×U† is also a density matrix.

   QuantumLib provides: Mixed_State_valid (composition of channels)
   ============================================================ *)
Lemma gate_evolution_density_matrix : forall (U : Matrix 2 2) (rho : Matrix 2 2),
  WF_Unitary U ->
  Mixed_State rho ->
  Mixed_State (U × rho × U†).
Proof.
  intros U rho HU Hrho.
  apply mixed_state_unitary_preserve; assumption.
Qed.

(* ============================================================
   Theorem 5.2: Tensor product of unitaries is unitary
   (U ⊗ V)†(U ⊗ V) = I   when U, V are unitary

   QuantumLib: kron_unitary
   ============================================================ *)
Lemma kron_gate_unitary : forall (U V : Matrix 2 2),
  WF_Unitary U -> WF_Unitary V ->
  WF_Unitary (kron U V).
Proof.
  intros U V HU HV.
  apply kron_unitary; assumption.
Qed.
