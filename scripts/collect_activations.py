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
    uv run --python 3.12 python scripts/collect_activations.py --run-name training_test_20260408_213832
"""

import argparse
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from simplexity.activations.activation_analyses import LinearRegressionAnalysis, PcaAnalysis
from simplexity.activations.activation_tracker import ActivationTracker
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import generate_data_batch_with_full_history
from simplexity.persistence.mlflow_persister import MLFlowPersister

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser()
_parser.add_argument("--run-name", type=str, default=None, help="MLflow run name (default: most recent run)")
_parser.add_argument("--experiment-name", type=str, default="/Shared/training_test")
_args = _parser.parse_args()

TRACKING_URI = "sqlite:///tests/end_to_end/mlflow.db"
EXPERIMENT_NAME: str = _args.experiment_name
RUN_NAME: str | None = _args.run_name

PROCESS_NAME = "mess3"
PROCESS_PARAMS = {"x": 0.15, "a": 0.6}
BOS_TOKEN = 3  # matches training config (base_vocab_size=3, bos_token=3)

VOCAB_SIZE = 3
SEQUENCE_LENGTH = 8  # matches training context length

OUTPUT_BASE_DIR = Path("outputs/activations")
PLOTS_BASE_DIR = Path("outputs/plots")

RESIDUAL_STREAM_HOOKS = [
    "hook_embed",
    "blocks.0.hook_resid_pre",
    "blocks.0.hook_resid_mid",
    "blocks.0.hook_resid_post",
    "ln_final.hook_normalized",
]

# ---------------------------------------------------------------------------
# Step 1: Find the MLflow run and list available checkpoints
# ---------------------------------------------------------------------------

print("Connecting to MLflow...")
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.MlflowClient(tracking_uri=TRACKING_URI)

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
assert experiment is not None, f"Experiment '{EXPERIMENT_NAME}' not found"

if RUN_NAME is not None:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{RUN_NAME}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
else:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

assert len(runs) > 0, f"No runs found in experiment '{EXPERIMENT_NAME}'"
run = runs[0]
run_id = run.info.run_id
run_name = run.info.run_name
print(f"Using run: {run_name} ({run_id})")

OUTPUT_DIR = OUTPUT_BASE_DIR / run_name
PLOTS_DIR = PLOTS_BASE_DIR / run_name

artifacts = client.list_artifacts(run_id, "models")
checkpoint_steps = sorted(int(a.path.split("/")[-1]) for a in artifacts if a.is_dir)
assert len(checkpoint_steps) > 0, "No checkpoint steps found in run artifacts"
print(f"Found {len(checkpoint_steps)} checkpoints at steps: {checkpoint_steps}")

# ---------------------------------------------------------------------------
# Step 2: Generate all possible sequences and compute belief states (once)
#
# Reuses generate_data_batch_with_full_history which handles:
#   - belief state computation via Bayesian filtering
#   - prefix probability computation
#   - bos token prepending
# We pass all sequences exhaustively rather than a random batch.
# ---------------------------------------------------------------------------

print(f"\nGenerating all {VOCAB_SIZE}^{SEQUENCE_LENGTH} = {VOCAB_SIZE**SEQUENCE_LENGTH} sequences...")

hmm = build_hidden_markov_model(process_name=PROCESS_NAME, process_params=PROCESS_PARAMS)

all_sequences = list(itertools.product(range(VOCAB_SIZE), repeat=SEQUENCE_LENGTH))
n_sequences = len(all_sequences)
sequences_jax = jnp.array(all_sequences)  # [n_seq, seq_len]

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

belief_states_np = np.array(belief_states)  # [n_seq, seq_len, n_states]
beliefs_flat = belief_states_np.reshape(-1, belief_states_np.shape[-1])

colors = beliefs_flat[:, :3]
colors = (colors - colors.min(axis=0)) / (colors.max(axis=0) - colors.min(axis=0) + 1e-8)

simplex_x = beliefs_flat[:, 1] - beliefs_flat[:, 0]
simplex_y = beliefs_flat[:, 2] - 0.5 * (beliefs_flat[:, 0] + beliefs_flat[:, 1])

print(f"Inputs shape: {np.array(inputs.cpu() if isinstance(inputs, torch.Tensor) else inputs).shape}")
print(f"Belief states shape: {belief_states_np.shape}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT_DIR / "inputs.npy", np.array(inputs))
np.save(OUTPUT_DIR / "belief_states.npy", belief_states_np)

# ---------------------------------------------------------------------------
# Step 3: For each checkpoint, load model, collect and save activations
# ---------------------------------------------------------------------------

persister = MLFlowPersister(
    experiment_name=EXPERIMENT_NAME,
    run_id=run_id,
    tracking_uri=TRACKING_URI,
)

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

from simplexity.analysis.metric_keys import format_layer_spec  # noqa: E402

for step in checkpoint_steps:
    print(f"\n{'=' * 60}")
    print(f"Checkpoint step {step}")
    print(f"{'=' * 60}")

    model = persister.load_model(step=step)
    model.eval()
    device = next(model.parameters()).device

    inputs_torch = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(np.array(inputs), dtype=torch.long)
    inputs_torch = inputs_torch.to(device)

    print("Running inference...")
    with torch.no_grad():
        _, act_cache = model.run_with_cache(inputs_torch)

    resid_acts = {k: v.detach().cpu() for k, v in act_cache.items() if k in RESIDUAL_STREAM_HOOKS}

    print("Running ActivationTracker analyses...")
    scalars, arrays = tracker.analyze(
        inputs=inputs_torch,
        beliefs=belief_states,
        probs=prefix_probs,
        activations=resid_acts,
    )

    print("Scalar results:")
    for key, value in scalars.items():
        print(f"  {key}: {value:.4f}")

    step_output_dir = OUTPUT_DIR / str(step)
    step_output_dir.mkdir(parents=True, exist_ok=True)
    step_plots_dir = PLOTS_DIR / str(step)
    step_plots_dir.mkdir(parents=True, exist_ok=True)

    for hook, activations in resid_acts.items():
        safe_hook = hook.replace(".", "_")
        np.save(step_output_dir / f"activations_{safe_hook}.npy", activations.numpy())

    for hook in RESIDUAL_STREAM_HOOKS:
        if hook not in resid_acts:
            continue
        safe_hook = hook.replace(".", "_")
        layer_spec = format_layer_spec(hook)

        pca_key = f"pca/pca/{layer_spec}"
        if pca_key not in arrays:
            pca_key = next((k for k in arrays if k.startswith("pca/pca")), None)
        if pca_key is None:
            print(f"  Skipping plot for {hook} (no PCA result)")
            continue

        projections = np.array(arrays[pca_key])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        r2_key = f"regression/r2/{layer_spec}"
        r2 = scalars.get(r2_key, float("nan"))
        fig.suptitle(f"Step {step} | Hook: {hook}  |  R²={r2:.4f}", fontsize=11)

        axes[0].scatter(projections[:, 0], projections[:, 1], c=colors, s=1, alpha=0.5)
        axes[0].set_title("Residual stream (PCA, colored by belief state)")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")

        axes[1].scatter(simplex_x, simplex_y, c=colors, s=1, alpha=0.5)
        axes[1].set_title("Ground-truth belief state geometry")
        axes[1].set_xlabel("Simplex x")
        axes[1].set_ylabel("Simplex y")

        plt.tight_layout()
        plot_path = step_plots_dir / f"{safe_hook}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"Saved activations to {step_output_dir}/")
    print(f"Saved plots to {step_plots_dir}/")

persister.cleanup()

print(f"\nDone. Processed {len(checkpoint_steps)} checkpoints.")
print("\nOutput structure:")
for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_dir():
        files = list(path.iterdir())
        print(f"  {path.name}/ ({len(files)} files)")
    else:
        size_mb = path.stat().st_size / 1e6
        print(f"  {path.name}: {size_mb:.2f} MB")
