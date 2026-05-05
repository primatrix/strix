#!/usr/bin/env python3
"""Generate Roofline model chart for TPU v7x per-TensorCore."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Hardware params (per TensorCore)
F_MXU = 1154  # TFLOPS BF16
BW_HBM = 3690  # GB/s
RIDGE = F_MXU * 1000 / BW_HBM  # FLOPs/byte ≈ 313

# AI range (log scale)
ai = np.logspace(-0.5, 3.5, 500)

# Roofline: min(F_MXU, AI * BW_HBM)
perf = np.minimum(F_MXU * 1000, ai * BW_HBM)  # in GFLOPS

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(ai, perf, 'k-', linewidth=2.5, label='Roofline (v7x per TC)')

# Ridge point
ax.axvline(x=RIDGE, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'Ridge = {RIDGE:.0f}', xy=(RIDGE, F_MXU * 1000 * 0.7),
            fontsize=9, color='gray', ha='right',
            xytext=(-10, 0), textcoords='offset points')

# Peak compute line
ax.axhline(y=F_MXU * 1000, color='gray', linestyle=':', alpha=0.3)
ax.annotate(f'F_MXU = {F_MXU} TFLOPS', xy=(3000, F_MXU * 1000),
            fontsize=9, color='gray', va='bottom')

# Working points
points = [
    (8, 'S2b Expert FFN\n(useful, decode)', 'red', 's', 12,
     f'AI=8, {8 * BW_HBM / 1000:.1f} TFLOPS\n({8 / RIDGE * 100:.1f}% peak)'),
    (32, 'S4 Shared Expert\n(bt=32)', 'blue', 'D', 10,
     f'AI=32, {32 * BW_HBM / 1000:.1f} TFLOPS\n({32 / RIDGE * 100:.1f}% peak)'),
    (67, 'S2b Expert FFN\n(exec, bt=64)', 'orange', '^', 11,
     f'AI=67, {67 * BW_HBM / 1000:.1f} TFLOPS\n({67 / RIDGE * 100:.1f}% peak)'),
]

for ai_val, label, color, marker, ms, detail in points:
    perf_val = min(F_MXU * 1000, ai_val * BW_HBM)
    ax.plot(ai_val, perf_val, marker=marker, color=color, markersize=ms,
            zorder=5, markeredgecolor='black', markeredgewidth=0.8)
    ax.annotate(f'{label}\n{detail}',
                xy=(ai_val, perf_val),
                xytext=(15, -10), textcoords='offset points',
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.8))

# Shade regions
ax.fill_between(ai[ai <= RIDGE], 0.1, perf[ai <= RIDGE],
                alpha=0.06, color='blue', label='HBM-bound region')
ax.fill_between(ai[ai >= RIDGE], 0.1, perf[ai >= RIDGE],
                alpha=0.06, color='red', label='Compute-bound region')

ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
ax.set_ylabel('Attainable Performance (GFLOPS)', fontsize=12)
ax.set_title('Roofline Model — TPU v7x per TensorCore (BF16)', fontsize=13)
ax.set_xlim(0.3, 3000)
ax.set_ylim(1, F_MXU * 1000 * 3)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, which='both', alpha=0.15)

plt.tight_layout()
plt.savefig('/Users/xl/Code/strix/docs/plans/images/roofline_v7x.png', dpi=150)
print("Saved roofline_v7x.png")
