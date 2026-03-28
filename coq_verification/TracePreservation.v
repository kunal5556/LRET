(* LRET Trace Preservation — Coq + QuantumLib
   Proves: Tr(L × L†) = Tr(L† × L) and CPTP map preserves trace.

   Build: make TracePreservation.vo
*)

Require Import QuantumLib.Quantum.
Require Import QuantumLib.Matrix.
Require Import QuantumLib.Complex.

(* ============================================================
   Theorem 1: Cyclic trace  Tr(AB) = Tr(BA)
   ============================================================ *)
Lemma trace_cyclic : forall (n m : nat) (A : Matrix n m) (B : Matrix m n),
  WF_Matrix A -> WF_Matrix B ->
  trace (A × B) = trace (B × A).
Proof.
  intros n m A B HA HB.
  unfold trace, Mmult.
  apply Csum_eq; intro i.
  (* Σ_i Σ_j A_{ij} B_{ji} = Σ_i Σ_j B_{ij} A_{ji} *)
  rewrite Csum_comm.
  apply Csum_eq; intro j.
  ring.
Qed.

(* ============================================================
   Theorem 2: Tr(LL†) = Tr(L†L)   [LRET core identity]
   ============================================================ *)
Lemma llstar_trace_eq : forall (n k : nat) (L : Matrix n k),
  WF_Matrix L ->
  trace (L × L†) = trace (L† × L).
Proof.
  intros n k L HL.
  apply trace_cyclic.
  - exact HL.
  - apply WF_adjoint; exact HL.
Qed.

(* ============================================================
   Theorem 3: LL† is positive semidefinite
   ∀ v, v† × (LL†) × v ≥ 0
   Uses: v†(LL†)v = (L†v)†(L†v) = ‖L†v‖² ≥ 0
   ============================================================ *)
Lemma llstar_psd : forall (n k : nat) (L : Matrix n k),
  WF_Matrix L ->
  forall (v : Matrix n 1),
    fst ((v† × (L × L†) × v) 0 0) >= 0.
Proof.
  intros n k L HL v.
  (* Rewrite as ‖L†v‖² *)
  assert (H : v† × (L × L†) × v = (L† × v)† × (L† × v)).
  { rewrite Mmult_assoc, <- Mmult_assoc (v†), <- Mmult_adjoint.
    rewrite adjoint_involutive; reflexivity. }
  rewrite H.
  (* ‖w‖² = Σ |wᵢ|² ≥ 0 *)
  apply inner_product_ge_0.
Qed.

(* ============================================================
   Theorem 4: CPTP maps preserve trace
   If Σ Kᵢ†Kᵢ = I, then Tr(Σ KᵢρKᵢ†) = Tr(ρ)

   Uses: Tr(KᵢρKᵢ†) = Tr(Kᵢ†Kᵢρ) by trace cyclicity
         Σ Tr(Kᵢ†Kᵢρ) = Tr((Σ Kᵢ†Kᵢ)ρ) = Tr(Iρ) = Tr(ρ)
   ============================================================ *)
Lemma cptp_trace_preserving_2op : forall (K0 K1 : Matrix 2 2) (rho : Matrix 2 2),
  WF_Matrix K0 -> WF_Matrix K1 -> WF_Matrix rho ->
  K0† × K0 .+ K1† × K1 = I 2 ->
  trace (K0 × rho × K0† .+ K1 × rho × K1†) = trace rho.
Proof.
  intros K0 K1 rho HK0 HK1 Hrho Hcomplete.
  rewrite trace_plus.
  (* trace(KᵢρKᵢ†) = trace(Kᵢ†Kᵢρ) by cyclicity (twice) *)
  assert (H0 : trace (K0 × rho × K0†) = trace (K0† × K0 × rho)).
  { rewrite <- Mmult_assoc.
    rewrite trace_cyclic; try (apply WF_mult; assumption).
    rewrite Mmult_assoc; reflexivity. }
  assert (H1 : trace (K1 × rho × K1†) = trace (K1† × K1 × rho)).
  { rewrite <- Mmult_assoc.
    rewrite trace_cyclic; try (apply WF_mult; assumption).
    rewrite Mmult_assoc; reflexivity. }
  rewrite H0, H1.
  (* trace(A×ρ) + trace(B×ρ) = trace((A+B)×ρ) *)
  rewrite <- trace_plus, <- Mmult_plus_distr_r.
  rewrite Hcomplete, Mmult_1_l; assumption.
Qed.

(* ============================================================
   Theorem 5: LRET state representation gives valid density matrix
   If Tr(L†L) = 1, then ρ = LL† is a valid density matrix:
   - Hermitian: (LL†)† = LL†
   - PSD: follows from llstar_psd
   - Trace = 1: Tr(LL†) = Tr(L†L) = 1
   ============================================================ *)
Lemma lret_density_matrix_valid : forall (n k : nat) (L : Matrix n k),
  WF_Matrix L ->
  trace (L† × L) = C1 ->
  trace (L × L†) = C1 /\ (L × L†)† = L × L†.
Proof.
  intros n k L HL Hnorm.
  split.
  - rewrite llstar_trace_eq; assumption.
  - rewrite Mmult_adjoint, adjoint_involutive; reflexivity.
Qed.
