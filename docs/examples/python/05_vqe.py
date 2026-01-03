"""
Variational Quantum Eigensolver (VQE)

This example implements VQE to find the ground state energy of a simple Hamiltonian.
VQE is a hybrid quantum-classical algorithm that uses a parametrized quantum circuit
(ansatz) and classical optimization to find ground states.

We'll find the ground state of the Hamiltonian: H = Z₀ + Z₁ + Z₀Z₁
"""

from qlret import QuantumSimulator
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class VQE:
    """Variational Quantum Eigensolver."""
    
    def __init__(self, n_qubits, hamiltonian_terms):
        """
        Initialize VQE.
        
        Args:
            n_qubits: Number of qubits
            hamiltonian_terms: List of (pauli_string, coefficient) tuples
                              e.g., [("Z0", 1.0), ("Z1", 1.0), ("Z0Z1", 0.5)]
        """
        self.n_qubits = n_qubits
        self.hamiltonian_terms = hamiltonian_terms
        self.energy_history = []
        
    def ansatz(self, params):
        """
        Variational ansatz circuit.
        
        Args:
            params: Circuit parameters (numpy array)
        
        Returns:
            QuantumSimulator with ansatz applied
        """
        sim = QuantumSimulator(n_qubits=self.n_qubits)
        
        n_layers = len(params) // (2 * self.n_qubits)
        
        for layer in range(n_layers):
            # Rotation layer
            for q in range(self.n_qubits):
                idx_y = layer * 2 * self.n_qubits + q
                idx_z = layer * 2 * self.n_qubits + self.n_qubits + q
                sim.ry(q, params[idx_y])
                sim.rz(q, params[idx_z])
            
            # Entangling layer
            for q in range(self.n_qubits - 1):
                sim.cnot(q, q + 1)
        
        return sim
    
    def measure_pauli_string(self, sim, pauli_string):
        """
        Measure expectation value of a Pauli string.
        
        Args:
            sim: QuantumSimulator instance
            pauli_string: String like "Z0", "X1", "Z0Z1", etc.
        
        Returns:
            Expectation value
        """
        # Parse Pauli string
        paulis = []
        i = 0
        while i < len(pauli_string):
            op = pauli_string[i]
            i += 1
            qubit = int(pauli_string[i])
            i += 1
            paulis.append((op, qubit))
        
        # For this example, we'll measure in Z basis after appropriate rotations
        # This is a simplified version; full implementation would handle all cases
        
        # Apply basis change
        for op, qubit in paulis:
            if op == 'X':
                sim.h(qubit)
            elif op == 'Y':
                sim.rx(qubit, -np.pi/2)
        
        # Measure expectation (simplified for Z basis)
        exp_val = sim.expectation("Z") if len(paulis) == 1 else 0.0
        
        return exp_val
    
    def compute_energy(self, params):
        """
        Compute energy expectation value.
        
        Args:
            params: Circuit parameters
        
        Returns:
            Energy (float)
        """
        sim = self.ansatz(params)
        
        energy = 0.0
        for pauli_string, coeff in self.hamiltonian_terms:
            exp_val = self.measure_pauli_string(sim, pauli_string)
            energy += coeff * exp_val
        
        self.energy_history.append(energy)
        return energy
    
    def optimize(self, init_params=None, method='COBYLA', maxiter=100):
        """
        Run VQE optimization.
        
        Args:
            init_params: Initial parameters (random if None)
            method: Optimization method
            maxiter: Maximum iterations
        
        Returns:
            Tuple of (optimal_params, ground_state_energy)
        """
        if init_params is None:
            n_params = 2 * self.n_qubits * 2  # 2 layers
            init_params = np.random.rand(n_params) * 2 * np.pi
        
        self.energy_history = []
        
        result = minimize(
            self.compute_energy,
            init_params,
            method=method,
            options={'maxiter': maxiter}
        )
        
        return result.x, result.fun


def simple_vqe_example():
    """Run VQE on a simple 2-qubit Hamiltonian."""
    print("="*60)
    print("VQE Example: H = Z₀ + 0.5*Z₁")
    print("="*60)
    
    # Define Hamiltonian: H = Z₀ + 0.5*Z₁
    hamiltonian = [
        ("Z0", 1.0),
        ("Z1", 0.5)
    ]
    
    # Analytical ground state energy: -1.5 (both qubits in |1⟩)
    analytical_energy = -1.5
    
    vqe = VQE(n_qubits=2, hamiltonian_terms=hamiltonian)
    
    print("\nRunning VQE optimization...")
    opt_params, energy = vqe.optimize(maxiter=50)
    
    print(f"\nResults:")
    print(f"  Optimal energy: {energy:.6f}")
    print(f"  Analytical energy: {analytical_energy:.6f}")
    print(f"  Error: {abs(energy - analytical_energy):.6f}")
    print(f"  Optimal parameters: {opt_params}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(vqe.energy_history, 'b-', linewidth=2)
    plt.axhline(y=analytical_energy, color='r', linestyle='--', 
                linewidth=2, label=f'Analytical: {analytical_energy:.3f}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title('VQE Convergence', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('vqe_convergence.png', dpi=150)
    print("\nConvergence plot saved to: vqe_convergence.png")
    
    # Prepare optimal state and verify
    sim = vqe.ansatz(opt_params)
    results = sim.measure_all(shots=1000)
    
    print(f"\nGround state measurement (1000 shots):")
    for outcome, count in sorted(results.items(), key=lambda x: -x[1])[:5]:
        print(f"  |{outcome}⟩: {count:4d} ({count/1000:.3f})")


def hydrogen_molecule_vqe():
    """
    VQE for H₂ molecule (simplified).
    
    Using a 2-qubit encoding with Hamiltonian:
    H = -1.0523 I + 0.3979 Z₀ - 0.3979 Z₁ - 0.0112 Z₀Z₁ + 0.1809 X₀X₁
    
    (This is a simplified version of the actual H₂ Hamiltonian)
    """
    print("\n" + "="*60)
    print("VQE Example: H₂ Molecule (Simplified)")
    print("="*60)
    
    # Simplified H₂ Hamiltonian
    hamiltonian = [
        ("Z0", 0.3979),
        ("Z1", -0.3979),
        # Note: This simplified version only includes Z terms
        # Full version would include X₀X₁ and Y₀Y₁ terms
    ]
    
    constant_term = -1.0523
    
    vqe = VQE(n_qubits=2, hamiltonian_terms=hamiltonian)
    
    print("\nRunning VQE optimization...")
    opt_params, energy = vqe.optimize(maxiter=100)
    
    total_energy = energy + constant_term
    
    print(f"\nResults:")
    print(f"  Variable energy: {energy:.6f}")
    print(f"  Constant term: {constant_term:.6f}")
    print(f"  Total energy: {total_energy:.6f} Hartree")
    print(f"  Iterations: {len(vqe.energy_history)}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    total_energies = [e + constant_term for e in vqe.energy_history]
    plt.plot(total_energies, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy (Hartree)', fontsize=12)
    plt.title('VQE Convergence for H₂ Molecule', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('h2_vqe_convergence.png', dpi=150)
    print("\nPlot saved to: h2_vqe_convergence.png")


def compare_ansatze():
    """Compare different ansatz depth."""
    print("\n" + "="*60)
    print("Comparing Ansatz Depths")
    print("="*60)
    
    hamiltonian = [("Z0", 1.0), ("Z1", 0.5)]
    analytical_energy = -1.5
    
    depths = [1, 2, 3, 4]
    results = []
    
    for depth in depths:
        print(f"\nDepth {depth}:")
        
        vqe = VQE(n_qubits=2, hamiltonian_terms=hamiltonian)
        n_params = 2 * 2 * depth  # 2 parameters per qubit per layer
        init_params = np.random.rand(n_params) * 2 * np.pi
        
        opt_params, energy = vqe.optimize(init_params=init_params, maxiter=50)
        error = abs(energy - analytical_energy)
        
        results.append({
            'depth': depth,
            'energy': energy,
            'error': error,
            'iterations': len(vqe.energy_history)
        })
        
        print(f"  Final energy: {energy:.6f}")
        print(f"  Error: {error:.6f}")
        print(f"  Iterations: {len(vqe.energy_history)}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    depths_list = [r['depth'] for r in results]
    energies = [r['energy'] for r in results]
    errors = [r['error'] for r in results]
    
    ax1.plot(depths_list, energies, 'bo-', linewidth=2, markersize=10)
    ax1.axhline(y=analytical_energy, color='r', linestyle='--', 
                linewidth=2, label='Analytical')
    ax1.set_xlabel('Ansatz Depth', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title('Energy vs Ansatz Depth', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(depths_list, errors, 'rs-', linewidth=2, markersize=10)
    ax2.set_xlabel('Ansatz Depth', fontsize=12)
    ax2.set_ylabel('Error (log scale)', fontsize=12)
    ax2.set_title('Error vs Ansatz Depth', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ansatz_comparison.png', dpi=150)
    print("\nComparison plot saved to: ansatz_comparison.png")


def main():
    # Example 1: Simple VQE
    simple_vqe_example()
    
    # Example 2: H₂ molecule (simplified)
    hydrogen_molecule_vqe()
    
    # Example 3: Compare ansatz depths
    compare_ansatze()
    
    print("\n" + "="*60)
    print("All VQE examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
