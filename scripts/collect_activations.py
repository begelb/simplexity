"""Collect residual stream activations and belief states from a trained transformer.

Reproduces the methodology from:
    "Transformers Represent Belief State Geometry in their Residual Stream"
    (arxiv 2405.15943)

Reuses existing simplexity infrastructure:
    - ActivationTracker for PCA and linear regression
    - generate_data_batch_with_full_history for belief state computation
    - MLFlowPersister for model loading

The two new operations not covered by existing code:
    - Exhaustive enumeration of all possible sequences (rather than random batches)
    - Saving raw activations to disk for persistent homology analysis

Usage:
    uv run --python 3.12 python scripts/collect_activations.py
"""

import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch

from simplexity.activations.activation_analyses import LinearRegressionAnalysis, PcaAnalysis
from simplexity.activations.activation_tracker import ActivationTracker
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import generate_data_batch_with_full_history

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRACKING_URI = "sqlite:///tests/end_to_end/mlflow.db"
MODEL_NAME = "hooked_transformer"
MODEL_VERSION = "2"

PROCESS_NAME = "mess3"
PROCESS_PARAMS = {"x": 0.15, "a": 0.6}
BOS_TOKEN = 3  # matches training config (base_vocab_size=3, bos_token=3)

VOCAB_SIZE = 3
SEQUENCE_LENGTH = 8  # matches training context length

OUTPUT_DIR = Path("outputs/activations")
PLOTS_DIR = Path("outputs/plots")

RESIDUAL_STREAM_HOOKS = [
    "hook_embed",
    "blocks.0.hook_resid_pre",
    "blocks.0.hook_resid_mid",
    "blocks.0.hook_resid_post",
    "ln_final.hook_normalized",
]

# ---------------------------------------------------------------------------
# Step 1: Load the trained model
# ---------------------------------------------------------------------------

print("Loading model from MLflow...")
mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
model.eval()
device = next(model.parameters()).device
print(f"Model loaded on device: {device}")

# ---------------------------------------------------------------------------
# Step 2: Generate all possible sequences and compute belief states
#
# Reuses generate_data_batch_with_full_history which handles:
#   - belief state computation via Bayesian filtering
#   - prefix probability computation
#   - bos token prepending
# We pass all sequences exhaustively rather than a random batch.
# ---------------------------------------------------------------------------

print(f"Generating all {VOCAB_SIZE}^{SEQUENCE_LENGTH} = {VOCAB_SIZE**SEQUENCE_LENGTH} sequences...")

hmm = build_hidden_markov_model(process_name=PROCESS_NAME, process_params=PROCESS_PARAMS)

all_sequences = list(itertools.product(range(VOCAB_SIZE), repeat=SEQUENCE_LENGTH))
n_sequences = len(all_sequences)
sequences_jax = jnp.array(all_sequences)  # [n_seq, seq_len]

# Expand initial state to match batch size
initial_states = jnp.repeat(hmm.initial_state[None, :], n_sequences, axis=0)  # [n_seq, n_states]

print("Computing belief states and prefix probabilities...")
outs = generate_data_batch_with_full_history(
    initial_states,
    hmm,
    n_sequences,
    SEQUENCE_LENGTH,
    jax.random.key(0),
    bos_token=BOS_TOKEN,
)

inputs = outs["inputs"]
belief_states = outs["belief_states"]
prefix_probs = outs["prefix_probabilities"]

assert isinstance(inputs, (jax.Array, torch.Tensor))
assert isinstance(belief_states, (jax.Array, tuple))
assert isinstance(prefix_probs, (jax.Array, torch.Tensor))

inputs_torch = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(np.array(inputs), dtype=torch.long)
inputs_torch = inputs_torch.to(device)

print(f"Inputs shape: {inputs_torch.shape}")
print(f"Belief states shape: {np.array(belief_states).shape if isinstance(belief_states, jax.Array) else 'factored'}")

# ---------------------------------------------------------------------------
# Step 3: Collect all residual stream activations
# ---------------------------------------------------------------------------

print("Running inference and collecting residual stream activations...")
with torch.no_grad():
    _, act_cache = model.run_with_cache(inputs_torch)

resid_acts = {k: v.detach().cpu() for k, v in act_cache.items() if k in RESIDUAL_STREAM_HOOKS}

print("Collected hook points:")
for k, v in resid_acts.items():
    print(f"  {k}: {v.shape}")

# ---------------------------------------------------------------------------
# Step 4: Run PCA and linear regression via ActivationTracker
#
# Reuses ActivationTracker which orchestrates both analyses with shared
# preprocessing (deduplication, token selection, weight computation).
# ---------------------------------------------------------------------------

print("\nRunning ActivationTracker analyses...")
tracker = ActivationTracker(
    analyses={
        "pca": PcaAnalysis(
            n_components=2,
            last_token_only=False,
            use_probs_as_weights=False,
            skip_deduplication=True,
        ),
        "regression": LinearRegressionAnalysis(
            last_token_only=False,
            use_probs_as_weights=False,
            skip_deduplication=True,
            fit_intercept=True,
        ),
    }
)

scalars, arrays = tracker.analyze(
    inputs=inputs_torch,
    beliefs=belief_states,
    probs=prefix_probs,
    activations=resid_acts,
)

print("\nRegression results per hook point:")
for key, value in scalars.items():
    if "r2" in key or "rmse" in key:
        print(f"  {key}: {value:.4f}")

# ---------------------------------------------------------------------------
# Step 5: Produce simplex projection plots
# ---------------------------------------------------------------------------

print("\nGenerating plots...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

belief_states_np = np.array(belief_states)  # [n_seq, seq_len, n_states]
beliefs_flat = belief_states_np.reshape(-1, belief_states_np.shape[-1])  # [n_seq * seq_len, n_states]

colors = beliefs_flat[:, :3]
colors = (colors - colors.min(axis=0)) / (colors.max(axis=0) - colors.min(axis=0) + 1e-8)

# Ground-truth simplex coordinates (2D projection of 3-simplex)
simplex_x = beliefs_flat[:, 1] - beliefs_flat[:, 0]
simplex_y = beliefs_flat[:, 2] - 0.5 * (beliefs_flat[:, 0] + beliefs_flat[:, 1])

for hook in RESIDUAL_STREAM_HOOKS:
    if hook not in resid_acts:
        continue
    safe_hook = hook.replace(".", "_")

    # PCA projections from ActivationTracker output
    pca_key = f"pca/{safe_hook}/pca"
    if pca_key not in arrays:
        # ActivationTracker concatenates layers — fall back to first available pca key
        pca_key = next((k for k in arrays if "pca" in k), None)
    if pca_key is None:
        print(f"  No PCA result found for {hook}, skipping plot")
        continue

    projections = np.array(arrays[pca_key])  # [n_seq * seq_len, 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    r2_key = f"regression/{safe_hook}/r2"
    r2 = scalars.get(r2_key, float("nan"))
    fig.suptitle(f"Hook: {hook}  |  R²={r2:.4f}", fontsize=11)

    axes[0].scatter(projections[:, 0], projections[:, 1], c=colors, s=1, alpha=0.5)
    axes[0].set_title("Residual stream (PCA, colored by belief state)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(simplex_x, simplex_y, c=colors, s=1, alpha=0.5)
    axes[1].set_title("Ground-truth belief state geometry")
    axes[1].set_xlabel("Simplex x")
    axes[1].set_ylabel("Simplex y")

    plt.tight_layout()
    plot_path = PLOTS_DIR / f"{safe_hook}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved {plot_path}")

# ---------------------------------------------------------------------------
# Step 6: Save raw data to disk for persistent homology analysis
# ---------------------------------------------------------------------------

print("\nSaving data to disk...")
np.save(OUTPUT_DIR / "inputs.npy", np.array(inputs_torch.cpu()))
np.save(OUTPUT_DIR / "belief_states.npy", belief_states_np)

for hook, activations in resid_acts.items():
    safe_hook = hook.replace(".", "_")
    np.save(OUTPUT_DIR / f"activations_{safe_hook}.npy", activations.numpy())

print("\nSaved files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    size_mb = path.stat().st_size / 1e6
    print(f"  {path.name}: {size_mb:.2f} MB")
