"""
Create comprehensive plots for LRET vs Cirq benchmark with noise
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open('results.json', 'r') as f:
    results = json.load(f)

qubits = [r['n_qubits'] for r in results]
lret_times = [r['lret_mean'] for r in results]
lret_stds = [r['lret_std'] for r in results]
cirq_times = [r['cirq_mean'] for r in results]
cirq_stds = [r['cirq_std'] for r in results]
speedups = [r['speedup'] for r in results]
ranks = [r['rank'] for r in results]

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'lret': '#2E86AB', 'cirq': '#A23B72', 'speedup': '#F18F01'}

# ============================================================================
# Plot 1: Side-by-side Time Comparison with Error Bars
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(qubits))
width = 0.35

bars1 = ax.bar(x - width/2, lret_times, width, yerr=lret_stds, 
               label='LRET (with DEPOLARIZE)', color=colors['lret'], 
               capsize=5, alpha=0.8)
bars2 = ax.bar(x + width/2, cirq_times, width, yerr=cirq_stds,
               label='Cirq (DensityMatrix)', color=colors['cirq'],
               capsize=5, alpha=0.8)

ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
ax.set_title('LRET vs Cirq: Execution Time with 0.1% Noise per Gate\n(Depth=15)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(qubits)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('execution_time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: execution_time_comparison.png")
plt.close()

# ============================================================================
# Plot 2: Speedup Chart with Annotations
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

line = ax.plot(qubits, speedups, 'o-', color=colors['speedup'], 
               linewidth=3, markersize=12, label='LRET Speedup')
ax.axhline(y=1, color='red', linestyle='--', linewidth=2, 
           alpha=0.7, label='Baseline (1x)')

ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
ax.set_ylabel('Speedup Factor', fontsize=13, fontweight='bold')
ax.set_title('LRET Speedup over Cirq (with Noise)\n0.1% Depolarizing per Gate, Depth=15', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate each point
for i, (q, s) in enumerate(zip(qubits, speedups)):
    ax.annotate(f'{s:.1f}x', 
                xy=(q, s), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=colors['speedup'], 
                         alpha=0.3))

plt.tight_layout()
plt.savefig('speedup_factor.png', dpi=300, bbox_inches='tight')
print("✓ Saved: speedup_factor.png")
plt.close()

# ============================================================================
# Plot 3: Rank Evolution
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(qubits, ranks, 'o-', color='#6A4C93', linewidth=3, 
        markersize=12, label='Final Rank')

ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
ax.set_ylabel('Final Rank (Entanglement)', fontsize=13, fontweight='bold')
ax.set_title('LRET Rank Evolution with Depolarizing Noise\n0.1% per Gate, Depth=15', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate each point
for q, r in zip(qubits, ranks):
    ax.annotate(f'{r}', 
                xy=(q, r), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=11,
                fontweight='bold')

plt.tight_layout()
plt.savefig('rank_evolution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rank_evolution.png")
plt.close()

# ============================================================================
# Plot 4: Combined Summary (2x2 grid)
# ============================================================================
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Time comparison
ax1 = plt.subplot(2, 2, 1)
ax1.plot(qubits, lret_times, 'o-', label='LRET', linewidth=2, markersize=8)
ax1.plot(qubits, cirq_times, 's-', label='Cirq', linewidth=2, markersize=8)
ax1.set_xlabel('Qubits', fontsize=11)
ax1.set_ylabel('Time (ms)', fontsize=11)
ax1.set_title('Execution Time', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Subplot 2: Speedup
ax2 = plt.subplot(2, 2, 2)
ax2.bar(qubits, speedups, color=colors['speedup'], alpha=0.7)
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Qubits', fontsize=11)
ax2.set_ylabel('Speedup (x)', fontsize=11)
ax2.set_title('LRET Speedup', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for i, (q, s) in enumerate(zip(qubits, speedups)):
    ax2.text(q, s + 0.1, f'{s:.1f}x', ha='center', va='bottom', fontweight='bold')

# Subplot 3: Rank
ax3 = plt.subplot(2, 2, 3)
ax3.plot(qubits, ranks, 'o-', color='purple', linewidth=2, markersize=8)
ax3.set_xlabel('Qubits', fontsize=11)
ax3.set_ylabel('Final Rank', fontsize=11)
ax3.set_title('Rank Evolution (Entanglement)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Subplot 4: Efficiency (time per qubit)
ax4 = plt.subplot(2, 2, 4)
lret_efficiency = [t / 2**q for t, q in zip(lret_times, qubits)]
cirq_efficiency = [t / 2**q for t, q in zip(cirq_times, qubits)]
ax4.plot(qubits, lret_efficiency, 'o-', label='LRET', linewidth=2, markersize=8)
ax4.plot(qubits, cirq_efficiency, 's-', label='Cirq', linewidth=2, markersize=8)
ax4.set_xlabel('Qubits', fontsize=11)
ax4.set_ylabel('Time / 2^n (ms)', fontsize=11)
ax4.set_title('Time per State (Efficiency)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

plt.suptitle('LRET vs Cirq: Comprehensive Benchmark with Noise\n' +
             '0.1% Depolarizing per Gate, Depth=15', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('comprehensive_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comprehensive_summary.png")
plt.close()

# ============================================================================
# Plot 5: Memory Efficiency (estimated)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Full density matrix: (2^n)^2 * 16 bytes (complex128) / 1e6 MB
full_memory = [(2**q)**2 * 16 / 1e6 for q in qubits]

# LRET with rank: rank * 2^n * 16 bytes / 1e6 MB
lret_memory = [r * 2**q * 16 / 1e6 for r, q in zip(ranks, qubits)]

ax.plot(qubits, full_memory, 's-', label='Full Density Matrix', 
        linewidth=2, markersize=8, color='red')
ax.plot(qubits, lret_memory, 'o-', label='LRET (Low-Rank)', 
        linewidth=2, markersize=8, color='green')

ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
ax.set_ylabel('Memory Usage (MB)', fontsize=13, fontweight='bold')
ax.set_title('Memory Efficiency: LRET vs Full Density Matrix', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Add memory reduction annotations
for q, full, lret in zip(qubits, full_memory, lret_memory):
    reduction = full / lret
    ax.annotate(f'{reduction:.0f}× less', 
                xy=(q, lret), 
                xytext=(10, -10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='green', alpha=0.2))

plt.tight_layout()
plt.savefig('memory_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: memory_efficiency.png")
plt.close()

print("\n" + "="*60)
print("All plots generated successfully!")
print("="*60)
