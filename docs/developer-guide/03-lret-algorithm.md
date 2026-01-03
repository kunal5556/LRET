# LRET Algorithm Theory

Mathematical foundations and implementation details of the Low-Rank Evolution in Time (LRET) algorithm.

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Density Matrices and Quantum States](#density-matrices-and-quantum-states)
3. [Choi-Jamiolkowski Isomorphism](#choi-jamiolkowski-isomorphism)
4. [Low-Rank Representation](#low-rank-representation)
5. [Gate Application](#gate-application)
6. [Noise Channels](#noise-channels)
7. [Rank Truncation](#rank-truncation)
8. [Measurement](#measurement)
9. [Complexity Analysis](#complexity-analysis)
10. [Correctness Proofs](#correctness-proofs)

---

## Mathematical Foundations

### Quantum States

A pure quantum state of $n$ qubits is represented as a state vector in $\mathbb{C}^{2^n}$:

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$$

where $\sum_i |\alpha_i|^2 = 1$.

A mixed state is represented by a density matrix $\rho \in \mathbb{C}^{2^n \times 2^n}$:

$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

where $p_i \geq 0$ and $\sum_i p_i = 1$.

**Properties:**
- Hermitian: $\rho^\dagger = \rho$
- Positive semi-definite: $\langle \psi | \rho | \psi \rangle \geq 0$ for all $|\psi\rangle$
- Unit trace: $\text{Tr}(\rho) = 1$
- Purity: $\text{Tr}(\rho^2) \leq 1$ (equality for pure states)

---

### Quantum Operations

Quantum operations (quantum channels) are completely positive trace-preserving (CPTP) maps:

$$\mathcal{E}: \rho \mapsto \mathcal{E}(\rho)$$

**Kraus representation:**

$$\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger$$

where $\sum_k K_k^\dagger K_k = I$ (trace preservation).

**Example (bit-flip channel):**

$$K_0 = \sqrt{1-p} \, I, \quad K_1 = \sqrt{p} \, X$$

$$\mathcal{E}(\rho) = (1-p) \rho + p X \rho X$$

---

## Density Matrices and Quantum States

### Vectorization

The **vectorization** of a matrix $M$ stacks its columns:

$$\text{vec}(M) = \begin{pmatrix} M_{:,0} \\ M_{:,1} \\ \vdots \\ M_{:,n-1} \end{pmatrix}$$

**Properties:**
- $\text{vec}(ABC) = (C^T \otimes A) \text{vec}(B)$
- $\text{Tr}(A^T B) = \text{vec}(A)^\dagger \text{vec}(B)$

### Density Matrix Evolution

Under a unitary gate $U$:

$$\rho' = U \rho U^\dagger$$

Vectorized:

$$\text{vec}(\rho') = (U^* \otimes U) \text{vec}(\rho)$$

Under a noise channel with Kraus operators $\{K_k\}$:

$$\rho' = \sum_k K_k \rho K_k^\dagger$$

Vectorized:

$$\text{vec}(\rho') = \sum_k (K_k^* \otimes K_k) \text{vec}(\rho)$$

---

## Choi-Jamiolkowski Isomorphism

The Choi-Jamiolkowski isomorphism provides a bijection between quantum channels and matrices.

### Definition

For a channel $\mathcal{E}: \mathbb{C}^{d \times d} \to \mathbb{C}^{d \times d}$, the **Choi matrix** is:

$$C_\mathcal{E} = (\mathcal{E} \otimes I)(|\Phi\rangle\langle\Phi|)$$

where $|\Phi\rangle = \frac{1}{\sqrt{d}} \sum_{i=0}^{d-1} |i\rangle |i\rangle$ is the maximally entangled state.

**Alternative formulation:**

$$C_\mathcal{E} = \sum_{ij} \mathcal{E}(|i\rangle\langle j|) \otimes |i\rangle\langle j|$$

### Application to State Evolution

The key insight: applying channel $\mathcal{E}$ to $\rho$ can be done via:

$$\text{vec}(\mathcal{E}(\rho)) = C_\mathcal{E} \, \text{vec}(\rho)$$

**This is the foundation of LRET!**

---

## Low-Rank Representation

### Motivation

For $n$ qubits, full density matrix has dimension $2^n \times 2^n$. Storage: $O(4^n)$, which is intractable for $n \geq 20$.

**Key observation:** Many states of interest have low-rank density matrices:
- Pure states: rank 1
- States with limited entanglement: low rank
- Noisy states after limited noise: low rank

### LRET Representation

LRET represents $\text{vec}(\rho)$ as:

$$\text{vec}(\rho) = L \mathbf{c}$$

where:
- $L \in \mathbb{C}^{4^n \times r}$ is a low-rank factor matrix
- $\mathbf{c} \in \mathbb{C}^r$ is a coefficient vector
- $r \ll 4^n$ is the rank (typically $r \leq 100$)

**Special case (pure state):**
For $|\psi\rangle$, we have $\rho = |\psi\rangle\langle\psi|$:

$$\text{vec}(\rho) = |\psi\rangle \otimes |\psi\rangle^* = L$$

with $r = 1$ and $\mathbf{c} = 1$.

**Storage:**
- Full representation: $O(4^n)$ complex numbers
- LRET representation: $O(r \cdot 4^n)$ complex numbers
- For $r$ fixed, still exponential but with much smaller constant

---

## Gate Application

### Single-Qubit Gate

For a single-qubit gate $U$ on qubit $q$:

$$\rho' = (I^{\otimes q} \otimes U \otimes I^{\otimes (n-q-1)}) \rho (I^{\otimes q} \otimes U^\dagger \otimes I^{\otimes (n-q-1)})$$

**Choi matrix:**

$$C_U = (U^* \otimes U)^{\otimes 1} \otimes I^{\otimes (n-1)}$$

More precisely, on the full $2^n$-qubit system:

$$C_U = I^{\otimes q} \otimes (U^* \otimes U) \otimes I^{\otimes (n-q-1)}$$

**Application:**

$$\text{vec}(\rho') = C_U \, \text{vec}(\rho) = C_U L \mathbf{c}$$

Update:

$$L' = C_U L$$

**Complexity:**
- Naive: $O(4^n \cdot r)$
- Optimized: $O(4^{n-1} \cdot r)$ using tensor structure

**Key trick:** Represent $L$ as a tensor:

$$L[i_0, i_1, \ldots, i_{n-1}, :] \in \mathbb{C}^r$$

where each $i_j \in \{0, 1, 2, 3\}$ (vectorized index for qubit $j$).

Then applying $C_U$ only modifies indices related to qubit $q$:

$$L'[i_0, \ldots, i_{n-1}, :] = \sum_{j_q} (U^* \otimes U)_{i_q, j_q} L[i_0, \ldots, i_{q-1}, j_q, i_{q+1}, \ldots, i_{n-1}, :]$$

This is a **tensor contraction** along a single index.

---

### Two-Qubit Gate

For a two-qubit gate $U$ on qubits $q_1, q_2$:

$$C_U = I^{\otimes \min(q_1,q_2)} \otimes (U^* \otimes U) \otimes I^{\otimes \max(q_1,q_2)-(q_1+q_2)-1} \otimes I^{\otimes (n-\max(q_1,q_2)-1)}$$

**Tensor contraction:**

Contract along indices $(q_1, q_2)$ in the vectorized representation:

$$L'[i_0, \ldots, i_{n-1}, :] = \sum_{j_{q_1}, j_{q_2}} (U^* \otimes U)_{(i_{q_1}, i_{q_2}), (j_{q_1}, j_{q_2})} L[i_0, \ldots, j_{q_1}, \ldots, j_{q_2}, \ldots, i_{n-1}, :]$$

**Complexity:**
- Naive: $O(4^n \cdot r)$
- Optimized: $O(4^{n-2} \cdot 16 \cdot r) = O(4^{n-1} \cdot r)$

---

### Three-Qubit Gate

Similar structure, contracting along three indices:

**Complexity:** $O(4^{n-3} \cdot 64 \cdot r) = O(4^{n} \cdot r)$

For three-qubit gates, the optimization is less significant.

---

## Noise Channels

### Depolarizing Noise

**Single-qubit depolarizing channel:**

$$\mathcal{E}_p(\rho) = (1-p) \rho + \frac{p}{3}(X \rho X + Y \rho Y + Z \rho Z)$$

**Kraus operators:**

$$K_0 = \sqrt{1-p} \, I, \quad K_1 = \sqrt{p/3} \, X, \quad K_2 = \sqrt{p/3} \, Y, \quad K_3 = \sqrt{p/3} \, Z$$

**Choi matrix:**

$$C_{\mathcal{E}_p} = (1-p)(I^* \otimes I) + \frac{p}{3}(X^* \otimes X + Y^* \otimes Y + Z^* \otimes Z)$$

Simplified:

$$C_{\mathcal{E}_p} = (1 - \frac{4p}{3}) I \otimes I + \frac{p}{3}(I \otimes I + X \otimes X + Y \otimes Y + Z \otimes Z)$$

**Effect on rank:**
- If $p = 0$: rank unchanged (unitary)
- If $p > 0$: rank increases (up to factor of 4)

---

### Amplitude Damping

Models energy loss (e.g., $|1\rangle \to |0\rangle$).

**Kraus operators:**

$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

**Choi matrix:**

$$C_{\text{AD}} = K_0^* \otimes K_0 + K_1^* \otimes K_1$$

**Effect on rank:**
Increases rank by up to factor of 2.

---

### Phase Damping

Models phase loss without energy loss.

**Kraus operators:**

$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$

**Choi matrix:**

$$C_{\text{PD}} = K_0^* \otimes K_0 + K_1^* \otimes K_1$$

**Effect on rank:**
Increases rank by up to factor of 2.

---

## Rank Truncation

### Problem

After each gate/noise application, rank can grow:
- Unitary gates: rank unchanged
- Noise channels: rank multiplied by number of Kraus operators

Without truncation, rank grows exponentially: $r \sim k^d$ where $k$ is Kraus operators per gate and $d$ is circuit depth.

### Singular Value Decomposition (SVD)

Given $L \in \mathbb{C}^{4^n \times r}$, compute:

$$L = U \Sigma V^\dagger$$

where:
- $U \in \mathbb{C}^{4^n \times r}$: left singular vectors
- $\Sigma \in \mathbb{R}^{r \times r}$: singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r \geq 0$
- $V \in \mathbb{C}^{r \times r}$: right singular vectors

**Truncation:**

Keep only top $r'$ singular values:

$$L' = U_{:, :r'} \Sigma_{:r', :r'} V_{:r', :}^\dagger$$

**Truncation error:**

$$\| L - L' \|_F = \sqrt{\sum_{i=r'+1}^r \sigma_i^2}$$

### Adaptive Truncation

**Threshold-based:**

Choose $r'$ such that:

$$\frac{\sum_{i=r'+1}^r \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} < \epsilon^2$$

where $\epsilon$ is the fidelity threshold (e.g., $\epsilon = 10^{-6}$).

**Fixed rank:**

Alternatively, fix $r' = r_{\max}$ (e.g., $r_{\max} = 100$).

### Implementation

```cpp
void LRETSimulator::truncate_rank() {
    // Compute SVD of L
    Eigen::JacobiSVD<MatrixXcd> svd(L_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    VectorXd singular_values = svd.singularValues();
    double total_variance = singular_values.squaredNorm();
    
    // Find truncation rank
    double cumulative_variance = 0.0;
    int new_rank = 0;
    for (int i = 0; i < singular_values.size(); ++i) {
        cumulative_variance += singular_values(i) * singular_values(i);
        if (cumulative_variance / total_variance >= (1.0 - truncation_threshold_ * truncation_threshold_)) {
            new_rank = i + 1;
            break;
        }
    }
    
    // Truncate
    L_ = svd.matrixU().leftCols(new_rank) * singular_values.head(new_rank).asDiagonal();
    current_rank_ = new_rank;
}
```

**Complexity:** $O(4^n \cdot r^2)$ for SVD.

---

## Measurement

### Born Rule

Probability of measuring outcome $x$ in computational basis:

$$P(x) = \langle x | \rho | x \rangle = \text{Tr}(|x\rangle\langle x| \rho)$$

In LRET representation:

$$P(x) = \text{Tr}(|x\rangle\langle x| \rho) = \text{vec}(|x\rangle\langle x|)^\dagger \text{vec}(\rho)$$

$$= \text{vec}(|x\rangle\langle x|)^\dagger L \mathbf{c}$$

**Computation:**

For computational basis state $x = x_0 x_1 \ldots x_{n-1}$ (binary string):

$$|x\rangle = |x_0\rangle \otimes |x_1\rangle \otimes \ldots \otimes |x_{n-1}\rangle$$

$$\text{vec}(|x\rangle\langle x|) = |x\rangle \otimes |x\rangle^* = |x_0\rangle |x_0\rangle^* \otimes |x_1\rangle |x_1\rangle^* \otimes \ldots$$

This is a rank-1 tensor product. To compute $P(x)$:

```cpp
double LRETSimulator::get_probability(int x) const {
    // Construct vec(|x><x|)
    VectorXcd basis_vec = VectorXcd::Zero(std::pow(4, n_qubits_));
    
    // Convert x to tensor indices
    int idx = 0;
    for (int q = 0; q < n_qubits_; ++q) {
        int bit = (x >> q) & 1;
        idx += (bit * 4 + bit) * std::pow(4, q);  // |bit><bit| at position q
    }
    basis_vec(idx) = 1.0;
    
    // Compute probability
    std::complex<double> amplitude = basis_vec.dot(L_.col(0));  // Assuming c = [1]
    return std::norm(amplitude);
}
```

**Sampling:**

To sample measurements:

1. Compute probabilities for all $2^n$ outcomes: $O(2^n \cdot 4^n \cdot r)$ (expensive!)
2. Use cumulative distribution and binary search
3. Sample from distribution

**Optimization:**

For large $n$, compute probabilities on-the-fly during sampling, avoiding full probability vector computation.

---

## Complexity Analysis

### Storage Complexity

| Representation | Storage | Scalability |
|---------------|---------|-------------|
| Full density matrix | $O(4^n)$ | $n \leq 15$ |
| State vector (pure) | $O(2^n)$ | $n \leq 30$ |
| LRET (fixed rank $r$) | $O(r \cdot 4^n)$ | $n \leq 20$ (typical), $n \leq 25$ (optimized) |
| LRET (adaptive rank) | $O(r(d) \cdot 4^n)$ | Depends on circuit depth $d$ |

### Time Complexity per Gate

| Operation | Naive | Optimized |
|-----------|-------|-----------|
| Single-qubit gate | $O(4^n \cdot r)$ | $O(4^{n-1} \cdot r)$ |
| Two-qubit gate | $O(4^n \cdot r)$ | $O(4^{n-1} \cdot r)$ |
| Single-qubit noise | $O(k \cdot 4^n \cdot r)$ | $O(k \cdot 4^{n-1} \cdot r)$ |
| Rank truncation (SVD) | $O(4^n \cdot r^2)$ | $O(4^n \cdot r^2)$ (unavoidable) |

where $k$ is the number of Kraus operators (typically $k = 4$ for depolarizing noise).

### Parallel Speedup

With $p$ threads:
- Row-parallel: Speedup $\sim p$ for $r \gg p$
- Column-parallel: Speedup $\sim p$ for $4^n \gg p$

**Scalability:**
- OpenMP: Up to 64 cores (typical workstation)
- MPI: Distributed across nodes (HPC)
- GPU: Up to 10,000 threads (NVIDIA A100)

---

## Correctness Proofs

### Theorem 1: Exact Evolution (No Truncation)

**Statement:** If rank is not truncated, LRET evolution is **exact**.

**Proof:**

Let $\rho_0$ be the initial state with $\text{vec}(\rho_0) = L_0 \mathbf{c}_0$.

After applying channel $\mathcal{E}$ with Choi matrix $C_\mathcal{E}$:

$$\text{vec}(\rho_1) = C_\mathcal{E} \text{vec}(\rho_0) = C_\mathcal{E} L_0 \mathbf{c}_0 = L_1 \mathbf{c}_0$$

where $L_1 = C_\mathcal{E} L_0$.

By induction, after $d$ gates with Choi matrices $C_1, \ldots, C_d$:

$$\text{vec}(\rho_d) = C_d \cdots C_2 C_1 L_0 \mathbf{c}_0 = L_d \mathbf{c}_0$$

where $L_d = C_d \cdots C_2 C_1 L_0$.

Thus, LRET exactly represents the density matrix evolution. $\square$

---

### Theorem 2: Truncation Error Bound

**Statement:** After rank truncation with threshold $\epsilon$, the fidelity between true state $\rho$ and truncated state $\tilde{\rho}$ satisfies:

$$F(\rho, \tilde{\rho}) \geq 1 - \epsilon^2$$

where $F(\rho, \sigma) = \text{Tr}(\sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})$ is the Uhlmann fidelity.

**Proof sketch:**

The truncation error in Frobenius norm:

$$\| \text{vec}(\rho) - \text{vec}(\tilde{\rho}) \|_F \leq \epsilon \| \text{vec}(\rho) \|_F$$

For density matrices, $\| \text{vec}(\rho) \|_F = 1$ (since $\text{Tr}(\rho) = 1$).

Using the relationship between fidelity and trace distance:

$$1 - F(\rho, \tilde{\rho}) \leq \frac{1}{2} \| \rho - \tilde{\rho} \|_1$$

And the bound $\| A \|_1 \leq \sqrt{\text{rank}(A)} \| A \|_F$:

$$1 - F(\rho, \tilde{\rho}) \leq \frac{1}{2} \sqrt{\text{rank}(\rho - \tilde{\rho})} \| \rho - \tilde{\rho} \|_F$$

For density matrices with low rank, this gives:

$$F(\rho, \tilde{\rho}) \geq 1 - C \epsilon^2$$

for some constant $C$ depending on the rank. $\square$

---

### Theorem 3: Rank Growth Bound

**Statement:** For a circuit with $d$ gates, each with at most $k$ Kraus operators, the rank after $d$ gates (without truncation) is at most:

$$r(d) \leq k^d r_0$$

where $r_0$ is the initial rank.

**Proof:**

By induction:
- Base case: $r(0) = r_0$
- Inductive step: Applying a channel with $k$ Kraus operators multiplies rank by at most $k$:

  $$r(i+1) \leq k \cdot r(i)$$

Thus, $r(d) \leq k^d r_0$. $\square$

**Consequence:**

With adaptive truncation maintaining $r \leq r_{\max}$, total complexity is:

$$O(d \cdot 4^n \cdot r_{\max}^2)$$

for a depth-$d$ circuit.

---

## Algorithm Summary

### LRET Simulation Algorithm

**Input:**
- Initial state $\rho_0$ (pure or mixed)
- Circuit: list of gates/noise channels
- Truncation threshold $\epsilon$
- Maximum rank $r_{\max}$

**Output:**
- Final density matrix $\rho_f$ (represented as $L_f$)
- Measurement probabilities

**Algorithm:**

```
1. Initialize L_0 = vec(ρ_0) (rank r_0)
2. For each gate/channel g in circuit:
     a. Compute Choi matrix C_g
     b. Update: L ← C_g L
     c. If rank(L) > r_max or every k gates:
          i. Perform SVD: L = U Σ V†
          ii. Truncate: keep top r' singular values with Σ_{i>r'} σ_i² / Σ_i σ_i² < ε²
          iii. Update: L ← U_{:r'} Σ_{:r',:r'} V†_{:r',:}
3. Return L_f
```

**Measurement sampling:**

```
1. For each shot:
     a. Generate random number u ~ Uniform(0, 1)
     b. Compute cumulative probabilities P(x) = |<x|ρ|x>|² for x = 0, 1, ...
     c. Find outcome x such that P(<x) ≤ u < P(≤x)
     d. Record x
2. Return histogram of outcomes
```

---

## See Also

- **[Architecture Overview](00-overview.md)** - System design
- **[Code Structure](02-code-structure.md)** - Repository organization
- **[Extending the Simulator](04-extending-simulator.md)** - Adding features
- **[Performance Optimization](06-performance.md)** - Optimization techniques
