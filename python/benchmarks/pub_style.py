"""
Shared IEEE/Nature publication style for LRET benchmark figures.
Apply by calling `apply_pub_style()` at the top of any benchmark script.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ──────────────────────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────────────────────
COLORS = {
    'lret':          '#2E86AB',   # blue
    'lret_opt':      '#2A9D8F',   # teal (optimized LRET)
    'cirq_fdm':      '#E76F51',   # orange-red (Cirq / full density matrix)
    'default_mixed': '#A23B72',   # magenta (PennyLane default.mixed)
    'lightning':     '#C73E1D',   # red (lightning.qubit)
    'qiskit':        '#8338EC',   # purple (Qiskit Aer)
    'baseline':      '#6C757D',   # grey (baseline / sequential)
    'phase1_2':      '#FFC300',   # amber
    'phase3_4':      '#FF5733',   # coral
    'full_opt':      '#2A9D8F',   # teal
    'oom_region':    '#FFCCCC',   # light red (OOM shading)
    'lret_region':   '#CCE5FF',   # light blue (LRET-only regime)
    'system_ram':    '#CC0000',   # dark red (RAM line)
}

# Figure sizes (inches)
FIGSIZE = {
    'single_col':  (3.5, 2.8),   # IEEE single-column
    'double_col':  (7.16, 5.0),  # IEEE double-column
    'double_tall': (7.16, 6.5),  # double-column, taller
    'square_sm':   (3.5, 3.5),   # small square
}

def apply_pub_style():
    """Apply IEEE/Nature publication rcParams globally."""
    matplotlib.rcParams.update({
        # Font
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size':          9,
        'axes.titlesize':     10,
        'axes.labelsize':     9,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'legend.fontsize':    8,
        'figure.titlesize':   11,
        # Lines
        'lines.linewidth':    1.5,
        'lines.markersize':   5,
        # Axes
        'axes.linewidth':     0.8,
        'axes.grid':          True,
        'grid.linewidth':     0.4,
        'grid.alpha':         0.5,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        # Figure
        'figure.dpi':         300,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.02,
        'figure.constrained_layout.use': True,
        # Math
        'mathtext.fontset':   'cm',
    })

def save_figure(fig, base_path: str):
    """Save figure as both PDF (vector) and PNG (300dpi)."""
    import pathlib
    p = pathlib.Path(base_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p) + '.pdf', format='pdf')
    fig.savefig(str(p) + '.png', format='png', dpi=300)
    print(f"  Saved: {p}.pdf + {p}.png")

def add_speedup_annotations(ax, x_vals, speedup_vals, threshold=1.0):
    """Shade region where speedup > threshold with light green."""
    for i, (x, s) in enumerate(zip(x_vals, speedup_vals)):
        if s > threshold:
            ax.axvspan(x - 0.3, x + 0.3, alpha=0.15, color='green', zorder=0)

def format_log_axis(ax, axis='y'):
    """Format a log-scale axis with clean major/minor tick labels."""
    if axis == 'y':
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    else:
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

def add_oom_region(ax, oom_x: float, x_max: float, y_min: float, y_max: float,
                   label: str = 'OOM Region'):
    """Shade the OOM region and add LRET-only region annotation."""
    ax.axvspan(oom_x, x_max, alpha=0.15, color=COLORS['oom_region'], label=label, zorder=0)
    ax.axvspan(oom_x, x_max, alpha=0.1, color=COLORS['lret_region'], zorder=0)

def make_heatmap_colormap():
    """Create a green-white-red diverging colormap for ratio heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#2DC653', '#FFFFFF', '#E63946']  # green=LRET wins, red=competitor
    return LinearSegmentedColormap.from_list('lret_ratio', colors_list, N=256)
