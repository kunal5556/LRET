"""
Noisy Circuit Simulation

This example demonstrates how to simulate quantum circuits with various types
of noise, including depolarizing, amplitude damping, and phase damping noise.
"""

from qlret import QuantumSimulator
import numpy as np
import matplotlib.pyplot as plt


def simulate_with_noise(n_qubits, depth, noise_level, noise_type='depolarizing'):
    """
    Simulate a circuit with noise.
    
    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        noise_level: Noise parameter
        noise_type: Type of noise ('depolarizing', 'amplitude_damping', 'phase_damping')
    
    Returns:
        Tuple of (sim, final_rank, fidelity_estimate)
    """
    sim = QuantumSimulator(n_qubits=n_qubits)
    
    # Apply a sequence of gates with noise
    for layer in range(depth):
        # Layer of Hadamards
        for q in range(n_qubits):
            sim.h(q)
            # Apply noise after each gate
            if noise_type == 'depolarizing':
                sim.apply_depolarizing_noise(q, noise_level)
            elif noise_type == 'amplitude_damping':
                sim.apply_amplitude_damping(q, noise_level)
            elif noise_type == 'phase_damping':
                sim.apply_phase_damping(q, noise_level)
        
        # Layer of CNOTs
        for q in range(0, n_qubits - 1, 2):
            sim.cnot(q, q + 1)
            # Apply two-qubit noise
            sim.apply_depolarizing_noise(q, noise_level * 2)
            sim.apply_depolarizing_noise(q + 1, noise_level * 2)
    
    return sim


def compare_noise_types():
    """Compare different noise types."""
    n_qubits = 4
    depth = 10
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05]
    noise_types = ['depolarizing', 'amplitude_damping', 'phase_damping']
    
    results = {noise_type: {'ranks': [], 'purities': []} for noise_type in noise_types}
    
    print("Comparing noise types...")
    print(f"Circuit: {n_qubits} qubits, depth {depth}\n")
    
    for noise_type in noise_types:
        print(f"\n{noise_type.upper()} NOISE:")
        print("-" * 50)
        
        for noise_level in noise_levels:
            sim = simulate_with_noise(n_qubits, depth, noise_level, noise_type)
            
            # Get state properties
            rho = sim.get_density_matrix()
            purity = np.trace(rho @ rho).real
            rank = sim.current_rank
            
            results[noise_type]['ranks'].append(rank)
            results[noise_type]['purities'].append(purity)
            
            print(f"  Noise level: {noise_level:.3f} → "
                  f"Rank: {rank:3d}, Purity: {purity:.4f}")
    
    # Visualize
    plot_noise_comparison(noise_levels, results)


def plot_noise_comparison(noise_levels, results):
    """Plot comparison of noise types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'depolarizing': 'blue', 'amplitude_damping': 'red', 'phase_damping': 'green'}
    markers = {'depolarizing': 'o', 'amplitude_damping': 's', 'phase_damping': '^'}
    
    # Plot ranks
    for noise_type, data in results.items():
        ax1.plot(noise_levels, data['ranks'], 
                marker=markers[noise_type], 
                color=colors[noise_type],
                label=noise_type.replace('_', ' ').title(),
                linewidth=2, markersize=8)
    
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Final Rank', fontsize=12)
    ax1.set_title('Rank Growth with Noise', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot purities
    for noise_type, data in results.items():
        ax2.plot(noise_levels, data['purities'],
                marker=markers[noise_type],
                color=colors[noise_type],
                label=noise_type.replace('_', ' ').title(),
                linewidth=2, markersize=8)
    
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Purity', fontsize=12)
    ax2.set_title('State Purity vs Noise', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('noise_comparison.png', dpi=150)
    print("\n\nPlot saved to: noise_comparison.png")


def fidelity_decay_experiment():
    """Study how fidelity decays with circuit depth."""
    n_qubits = 4
    max_depth = 50
    noise_level = 0.01
    
    depths = range(0, max_depth + 1, 5)
    fidelities = []
    
    print("\n" + "="*60)
    print("Fidelity Decay Experiment")
    print("="*60)
    print(f"Qubits: {n_qubits}, Noise level: {noise_level}")
    print()
    
    for depth in depths:
        # Noiseless simulation
        sim_ideal = QuantumSimulator(n_qubits=n_qubits)
        
        # Noisy simulation
        sim_noisy = QuantumSimulator(n_qubits=n_qubits)
        
        # Apply same circuit to both
        for layer in range(depth):
            for q in range(n_qubits):
                sim_ideal.h(q)
                sim_noisy.h(q)
                sim_noisy.apply_depolarizing_noise(q, noise_level)
            
            for q in range(0, n_qubits - 1, 2):
                sim_ideal.cnot(q, q + 1)
                sim_noisy.cnot(q, q + 1)
                sim_noisy.apply_depolarizing_noise(q, noise_level * 2)
                sim_noisy.apply_depolarizing_noise(q + 1, noise_level * 2)
        
        # Compute fidelity
        rho_ideal = sim_ideal.get_density_matrix()
        rho_noisy = sim_noisy.get_density_matrix()
        
        # Fidelity: F = Tr(ρ_ideal @ ρ_noisy)
        fidelity = np.trace(rho_ideal @ rho_noisy).real
        fidelities.append(fidelity)
        
        print(f"Depth {depth:3d}: Fidelity = {fidelity:.6f}, "
              f"Noisy rank = {sim_noisy.current_rank:3d}")
    
    # Plot fidelity decay
    plt.figure(figsize=(10, 6))
    plt.plot(depths, fidelities, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Circuit Depth', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title(f'Fidelity Decay with Noise (p={noise_level})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Fit exponential decay
    from scipy.optimize import curve_fit
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)
    
    try:
        popt, _ = curve_fit(exp_decay, depths, fidelities, p0=[1.0, 0.01])
        fit_y = exp_decay(np.array(depths), *popt)
        plt.plot(depths, fit_y, 'r--', linewidth=2, 
                label=f'Fit: F = {popt[0]:.3f} exp(-{popt[1]:.4f} × depth)')
        plt.legend(fontsize=10)
    except:
        pass
    
    plt.tight_layout()
    plt.savefig('fidelity_decay.png', dpi=150)
    print("\nPlot saved to: fidelity_decay.png")


def main():
    # Experiment 1: Compare noise types
    compare_noise_types()
    
    # Experiment 2: Fidelity decay
    fidelity_decay_experiment()
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)


if __name__ == "__main__":
    main()
