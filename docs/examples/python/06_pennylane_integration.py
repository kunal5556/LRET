"""
PennyLane Integration Example

This example demonstrates how to use LRET as a PennyLane device for
variational quantum algorithms and automatic differentiation.
"""

import pennylane as qml
from qlret import QLRETDevice
import numpy as np
import matplotlib.pyplot as plt


def basic_pennylane_example():
    """Basic PennyLane circuit with LRET device."""
    print("="*60)
    print("Basic PennyLane Example with LRET Device")
    print("="*60)
    
    # Create LRET device
    dev = qml.device("qlret.simulator", wires=2, noise_level=0.0)
    
    @qml.qnode(dev)
    def circuit(theta):
        """Simple parametric circuit."""
        qml.Hadamard(wires=0)
        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])
    
    # Run circuit
    theta = np.pi / 4
    probs = circuit(theta)
    
    print(f"\nCircuit with θ = π/4:")
    print(f"  Probabilities: {probs}")
    print(f"  Sum: {np.sum(probs):.6f} (should be 1.0)")


def gradient_computation_example():
    """Demonstrate automatic differentiation."""
    print("\n" + "="*60)
    print("Gradient Computation with Parameter-Shift Rule")
    print("="*60)
    
    dev = qml.device("qlret.simulator", wires=1)
    
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    # Compute expectation and gradient
    theta = 0.5
    expectation = circuit(theta)
    gradient = qml.grad(circuit)(theta)
    
    # Analytical values
    analytical_exp = np.cos(theta)
    analytical_grad = -np.sin(theta)
    
    print(f"\nθ = {theta:.4f}:")
    print(f"  Expectation value: {expectation:.6f}")
    print(f"  Analytical: {analytical_exp:.6f}")
    print(f"  Error: {abs(expectation - analytical_exp):.8f}")
    print(f"\n  Gradient: {gradient:.6f}")
    print(f"  Analytical: {analytical_grad:.6f}")
    print(f"  Error: {abs(gradient - analytical_grad):.8f}")


def vqe_with_pennylane():
    """VQE using PennyLane with LRET device."""
    print("\n" + "="*60)
    print("VQE with PennyLane + LRET")
    print("="*60)
    
    # Define Hamiltonian: H = 0.5 * Z₀ + 0.3 * Z₁
    coeffs = [0.5, 0.3]
    obs = [qml.PauliZ(0), qml.PauliZ(1)]
    H = qml.Hamiltonian(coeffs, obs)
    
    # Create device
    dev = qml.device("qlret.simulator", wires=2)
    
    @qml.qnode(dev)
    def cost_fn(params):
        """Cost function (energy)."""
        # Ansatz
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[2], wires=0)
        qml.RY(params[3], wires=1)
        
        return qml.expval(H)
    
    # Optimize
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    params = np.random.rand(4) * np.pi
    
    energies = []
    print("\nOptimization:")
    for i in range(50):
        params = opt.step(cost_fn, params)
        energy = cost_fn(params)
        energies.append(energy)
        
        if i % 10 == 0:
            print(f"  Step {i:2d}: Energy = {energy:.6f}")
    
    final_energy = cost_fn(params)
    analytical_min = -0.8  # Ground state energy
    
    print(f"\nFinal Results:")
    print(f"  Final energy: {final_energy:.6f}")
    print(f"  Analytical minimum: {analytical_min:.6f}")
    print(f"  Error: {abs(final_energy - analytical_min):.6f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(energies, 'b-', linewidth=2)
    plt.axhline(y=analytical_min, color='r', linestyle='--', 
                linewidth=2, label=f'Ground state: {analytical_min:.3f}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title('VQE Convergence (PennyLane + LRET)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pennylane_vqe.png', dpi=150)
    print("\nPlot saved to: pennylane_vqe.png")


def qaoa_example():
    """QAOA (Quantum Approximate Optimization Algorithm) example."""
    print("\n" + "="*60)
    print("QAOA with PennyLane + LRET")
    print("="*60)
    
    # Simple MaxCut problem on a 3-node graph
    # Graph edges: (0,1), (1,2), (0,2) - triangle
    
    n_qubits = 3
    dev = qml.device("qlret.simulator", wires=n_qubits)
    
    # Cost Hamiltonian (MaxCut)
    def cost_hamiltonian(wires):
        """Cost Hamiltonian for MaxCut."""
        # Apply ZZ interactions for each edge
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RZ(-0.5, wires=wires[1])  # Half angle for ZZ
        qml.CNOT(wires=[wires[0], wires[1]])
        
        qml.CNOT(wires=[wires[1], wires[2]])
        qml.RZ(-0.5, wires=wires[2])
        qml.CNOT(wires=[wires[1], wires[2]])
        
        qml.CNOT(wires=[wires[0], wires[2]])
        qml.RZ(-0.5, wires=wires[2])
        qml.CNOT(wires=[wires[0], wires[2]])
    
    def mixer_hamiltonian(wires):
        """Mixer Hamiltonian (X on all qubits)."""
        for wire in wires:
            qml.RX(1.0, wires=wire)
    
    @qml.qnode(dev)
    def qaoa_circuit(gamma, beta):
        """QAOA circuit."""
        wires = range(n_qubits)
        
        # Initial state: uniform superposition
        for wire in wires:
            qml.Hadamard(wires=wire)
        
        # QAOA layers
        cost_hamiltonian(wires)
        mixer_hamiltonian(wires)
        
        # Measurement
        return qml.probs(wires=wires)
    
    # Run with optimal parameters (found via optimization)
    gamma_opt = 0.7
    beta_opt = 0.4
    
    probs = qaoa_circuit(gamma_opt, beta_opt)
    
    print("\nQAOA Results:")
    print("  Outcome probabilities:")
    for i, prob in enumerate(probs):
        if prob > 0.05:  # Show only significant outcomes
            bitstring = format(i, f'0{n_qubits}b')
            print(f"    |{bitstring}⟩: {prob:.4f}")
    
    # MaxCut value
    print("\n  Expected cuts:")
    for i, prob in enumerate(probs):
        bitstring = format(i, f'0{n_qubits}b')
        # Count edges cut by this partition
        cuts = 0
        if bitstring[0] != bitstring[1]: cuts += 1
        if bitstring[1] != bitstring[2]: cuts += 1
        if bitstring[0] != bitstring[2]: cuts += 1
        if prob > 0.05:
            print(f"    |{bitstring}⟩: {cuts} edges cut (prob {prob:.4f})")


def noisy_device_example():
    """Example with noisy LRET device."""
    print("\n" + "="*60)
    print("Noisy Simulation with PennyLane")
    print("="*60)
    
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    results = []
    
    for noise in noise_levels:
        dev = qml.device("qlret.simulator", wires=2, noise_level=noise)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])
        
        probs = circuit()
        results.append(probs)
        
        print(f"\nNoise level: {noise:.2f}")
        print(f"  |00⟩: {probs[0]:.4f}")
        print(f"  |01⟩: {probs[1]:.4f}")
        print(f"  |10⟩: {probs[2]:.4f}")
        print(f"  |11⟩: {probs[3]:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(16, 4))
    
    for idx, (noise, probs) in enumerate(zip(noise_levels, results)):
        ax = axes[idx]
        ax.bar(['00', '01', '10', '11'], probs, color='steelblue', edgecolor='black')
        ax.set_title(f'Noise: {noise:.2f}', fontsize=12)
        ax.set_ylabel('Probability' if idx == 0 else '', fontsize=10)
        ax.set_ylim([0, 0.6])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pennylane_noise.png', dpi=150)
    print("\nPlot saved to: pennylane_noise.png")


def main():
    # Example 1: Basic PennyLane usage
    basic_pennylane_example()
    
    # Example 2: Gradient computation
    gradient_computation_example()
    
    # Example 3: VQE
    vqe_with_pennylane()
    
    # Example 4: QAOA
    qaoa_example()
    
    # Example 5: Noisy simulation
    noisy_device_example()
    
    print("\n" + "="*60)
    print("All PennyLane examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
