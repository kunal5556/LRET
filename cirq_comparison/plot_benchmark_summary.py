"""
Create a quick visualization of LRET vs Cirq benchmark results
"""

import matplotlib.pyplot as plt
import numpy as np

# Results data from benchmark
circuits = ['Bell\n2q', 'Bell\n4q', 'Bell\n6q', 'GHZ\n3q', 'GHZ\n4q', 'GHZ\n6q',
            'QFT\n3q', 'QFT\n4q', 'QFT\n5q', 'QFT\n6q', 'Rand\n4q', 'Rand\n6q']
categories = ['Low-Rank'] * 6 + ['Moderate'] * 4 + ['High-Rank'] * 2
speedups = [2.26, 28.40, 88.14, 35.31, 39.45, 101.33, 97.91, 122.27, 156.86, 151.18, 157.93, 186.53]
cirq_times = [3.60, 5.50, 7.89, 5.12, 5.95, 8.05, 10.65, 17.36, 28.06, 40.83, 34.92, 58.47]
lret_times = [1.59, 0.19, 0.09, 0.14, 0.15, 0.08, 0.11, 0.14, 0.18, 0.27, 0.22, 0.31]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Speedup comparison
colors = ['#3498db' if cat == 'Low-Rank' else '#2ecc71' if cat == 'Moderate' else '#e74c3c' 
          for cat in categories]
bars = ax1.bar(range(len(circuits)), speedups, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Baseline (1×)')
ax1.set_xlabel('Circuit', fontsize=11, fontweight='bold')
ax1.set_ylabel('Speedup (LRET vs Cirq)', fontsize=11, fontweight='bold')
ax1.set_title('LRET Speedup Over Cirq FDM\nAverage: 97.30×', fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(circuits)))
ax1.set_xticklabels(circuits, rotation=0, ha='center', fontsize=9)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.legend(['Low-Rank (49×)', 'Moderate (132×)', 'High-Rank (172×)'], loc='upper left')

# Add speedup values on bars
for i, (bar, speedup) in enumerate(zip(bars, speedups)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height*1.1,
             f'{speedup:.0f}×',
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 2: Execution time comparison
x = np.arange(len(circuits))
width = 0.35
bars1 = ax2.bar(x - width/2, cirq_times, width, label='Cirq FDM', color='#e74c3c', alpha=0.7)
bars2 = ax2.bar(x + width/2, lret_times, width, label='LRET', color='#2ecc71', alpha=0.7)

ax2.set_xlabel('Circuit', fontsize=11, fontweight='bold')
ax2.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
ax2.set_title('Execution Time Comparison\nTotal: Cirq 226ms, LRET 3ms', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(circuits, rotation=0, ha='center', fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('d:/LRET/cirq_comparison/benchmark_summary.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved to: d:/LRET/cirq_comparison/benchmark_summary.png")
plt.show()

print("\n" + "="*60)
print("LRET vs Cirq Benchmark Summary")
print("="*60)
print(f"Average Speedup: 97.30×")
print(f"Best Speedup: {max(speedups):.2f}× ({circuits[speedups.index(max(speedups))].replace(chr(10), ' ')})")
print(f"Total Time - Cirq: {sum(cirq_times):.0f} ms")
print(f"Total Time - LRET: {sum(lret_times):.2f} ms")
print(f"Time Saved: {sum(cirq_times) - sum(lret_times):.0f} ms ({100*(1-sum(lret_times)/sum(cirq_times)):.1f}%)")
print("="*60)
