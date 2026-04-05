"""Persistent homology analysis of belief state geometry and residual stream activations.

Loads saved numpy arrays from scripts/collect_activations.py and computes
Vietoris-Rips persistent homology on:
    1. Ground-truth belief states (the true geometry in the 2-simplex)
    2. Residual stream activations at each hook point (the transformer's learned geometry)

Persistence diagrams are saved to outputs/plots/homology/.

Dependencies (install locally):
    pip install ripser persim matplotlib numpy

Usage:
    python scripts/persistent_homology.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import bottleneck, plot_diagrams, wasserstein
from ripser import ripser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = Path("outputs/activations")
PLOTS_DIR = Path("outputs/plots/homology")

# Use last token position only — cleanest signal, matches paper's approach
LAST_TOKEN_ONLY = True

# Homology dimensions to compute (0=connected components, 1=loops, 2=voids)
MAX_DIM = 2

# Subsample if needed — ripser scales as O(n^3), so limit point count
MAX_POINTS = 2000

HOOK_NAMES = [
    "hook_embed",
    "blocks_0_hook_resid_pre",
    "blocks_0_hook_resid_mid",
    "blocks_0_hook_resid_post",
    "ln_final_hook_normalized",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Randomly subsample a point cloud if it exceeds max_points."""
    if len(points) <= max_points:
        return points
    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx]


def run_and_plot(points: np.ndarray, title: str, save_path: Path) -> list[np.ndarray]:
    """Run ripser on a point cloud, save the persistence diagram, and return diagrams."""
    points = subsample(points, MAX_POINTS)
    print(f"  Running ripser on {len(points)} points...")
    result = ripser(points, maxdim=MAX_DIM)
    diagrams = result["dgms"]

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_diagrams(diagrams, ax=ax, show=False)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")

    for dim, dgm in enumerate(diagrams):
        finite = dgm[~np.isinf(dgm[:, 1])]
        persistence = finite[:, 1] - finite[:, 0]
        if len(persistence) > 0:
            print(f"    H{dim}: {len(finite)} finite bars, max persistence={persistence.max():.4f}")

    return diagrams


def compare_diagrams(
    dgms_learned: list[np.ndarray],
    dgms_true: list[np.ndarray],
    label: str,
) -> dict[str, float]:
    """Compute Wasserstein and bottleneck distances between two persistence diagrams.

    Compares learned (transformer) diagrams against ground-truth (belief state)
    diagrams per homology dimension. Lower distance = closer topology.
    """
    results = {}
    for dim in range(min(MAX_DIM + 1, len(dgms_learned), len(dgms_true))):
        w_dist = wasserstein(dgms_learned[dim], dgms_true[dim])
        b_dist = bottleneck(dgms_learned[dim], dgms_true[dim])
        results[f"H{dim}_wasserstein"] = float(w_dist)
        results[f"H{dim}_bottleneck"] = float(b_dist)
        print(f"    H{dim}: Wasserstein={w_dist:.4f}  Bottleneck={b_dist:.4f}")
    return results


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading saved data...")
belief_states = np.load(INPUT_DIR / "belief_states.npy")  # [n_seq, seq_len, n_states]
print(f"Belief states shape: {belief_states.shape}")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Persistent homology on ground-truth belief states
# ---------------------------------------------------------------------------

print("\n--- Ground-truth belief states ---")
if LAST_TOKEN_ONLY:
    belief_points = belief_states[:, -1, :]  # [n_seq, n_states]
else:
    belief_points = belief_states.reshape(-1, belief_states.shape[-1])  # [n_seq * seq_len, n_states]

print(f"  Point cloud shape: {belief_points.shape}")
belief_diagrams = run_and_plot(
    belief_points,
    title="Persistent homology — ground-truth belief states",
    save_path=PLOTS_DIR / "belief_states.png",
)

# ---------------------------------------------------------------------------
# 2. Persistent homology on residual stream activations per hook point
# ---------------------------------------------------------------------------

comparison: dict[str, dict[str, float]] = {}

for hook in HOOK_NAMES:
    activation_path = INPUT_DIR / f"activations_{hook}.npy"
    if not activation_path.exists():
        print(f"\nSkipping {hook} (file not found)")
        continue

    print(f"\n--- {hook} ---")
    activations = np.load(activation_path)  # [n_seq, seq_len, d_model]

    if LAST_TOKEN_ONLY:
        points = activations[:, -1, :]  # [n_seq, d_model]
    else:
        points = activations.reshape(-1, activations.shape[-1])  # [n_seq * seq_len, d_model]

    print(f"  Point cloud shape: {points.shape}")
    learned_diagrams = run_and_plot(
        points,
        title=f"Persistent homology — {hook}",
        save_path=PLOTS_DIR / f"{hook}.png",
    )

    print(f"  Comparing to ground-truth belief states:")
    comparison[hook] = compare_diagrams(learned_diagrams, belief_diagrams, label=hook)

# ---------------------------------------------------------------------------
# 3. Save comparison results
# ---------------------------------------------------------------------------

print("\n--- Diagram distance summary ---")
rows = []
for hook, metrics in comparison.items():
    row = {"hook": hook} | metrics
    rows.append(row)
    print(f"  {hook}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

df = pd.DataFrame(rows).set_index("hook")
csv_path = PLOTS_DIR / "diagram_distances.csv"
df.to_csv(csv_path)
print(f"\nSaved distance summary to {csv_path}")

print("\nDone.")
