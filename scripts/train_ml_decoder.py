#!/usr/bin/env python3
"""
Phase 9.3: ML Decoder Training Script

Trains a Transformer-based neural network decoder for quantum error correction
using JAX/Flax. The trained model can be exported for use with the C++ runtime.

Usage:
    python scripts/train_ml_decoder.py --train-data train_data.npz \
        --val-data train_data_val.npz --output models/decoder_surface_d5.pkl

Requirements:
    pip install jax jaxlib flax optax numpy
"""

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax.training import train_state, checkpoints
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: JAX/Flax not available. Install with: pip install jax jaxlib flax optax")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the Transformer decoder model."""

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    num_classes: int = 4  # I, X, Z, Y


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 256
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    num_epochs: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    early_stopping_patience: int = 10


if JAX_AVAILABLE:

    class TransformerBlock(nn.Module):
        """Single Transformer block with self-attention and MLP."""

        hidden_dim: int
        num_heads: int
        mlp_dim: int
        dropout_rate: float = 0.1

        @nn.compact
        def __call__(self, x, train: bool = True):
            # Self-attention with residual connection
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_dim,
                dropout_rate=self.dropout_rate,
            )(x, x, deterministic=not train)

            x = nn.LayerNorm()(x + attn_output)

            # MLP with residual connection
            mlp_output = nn.Dense(self.mlp_dim)(x)
            mlp_output = nn.gelu(mlp_output)
            mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(mlp_output)
            mlp_output = nn.Dense(self.hidden_dim)(mlp_output)
            mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(mlp_output)

            x = nn.LayerNorm()(x + mlp_output)
            return x


    class TransformerDecoder(nn.Module):
        """Transformer-based QEC decoder."""

        config: ModelConfig
        num_qubits: int
        syndrome_size: int

        @nn.compact
        def __call__(self, syndrome, train: bool = True):
            """
            Args:
                syndrome: Input syndrome of shape (batch, syndrome_size)

            Returns:
                logits: Per-qubit error logits of shape (batch, num_qubits, 4)
            """
            batch_size = syndrome.shape[0]

            # Embed syndrome bits
            x = nn.Dense(self.config.hidden_dim)(syndrome.astype(jnp.float32))
            x = nn.gelu(x)

            # Reshape for attention: (batch, seq_len=1, hidden)
            x = x[:, None, :]

            # Add learnable position embedding
            pos_embed = self.param(
                "pos_embed",
                nn.initializers.normal(stddev=0.02),
                (1, 1, self.config.hidden_dim),
            )
            x = x + pos_embed

            # Transformer blocks
            for i in range(self.config.num_layers):
                x = TransformerBlock(
                    hidden_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    mlp_dim=self.config.mlp_dim,
                    dropout_rate=self.config.dropout_rate,
                    name=f"transformer_block_{i}",
                )(x, train=train)

            # Flatten and project to output
            x = x.reshape(batch_size, -1)

            # Output heads for each qubit
            logits = nn.Dense(self.num_qubits * self.config.num_classes)(x)
            logits = logits.reshape(batch_size, self.num_qubits, self.config.num_classes)

            return logits


    class MLPDecoder(nn.Module):
        """Simple MLP-based decoder for comparison."""

        hidden_dims: List[int]
        num_qubits: int
        num_classes: int = 4
        dropout_rate: float = 0.1

        @nn.compact
        def __call__(self, syndrome, train: bool = True):
            x = syndrome.astype(jnp.float32)

            for i, dim in enumerate(self.hidden_dims):
                x = nn.Dense(dim)(x)
                x = nn.LayerNorm()(x)
                x = nn.gelu(x)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

            logits = nn.Dense(self.num_qubits * self.num_classes)(x)
            logits = logits.reshape(-1, self.num_qubits, self.num_classes)
            return logits


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load training data from .npz file."""
    data = np.load(path, allow_pickle=True)
    syndromes = data["syndromes"]
    errors = data["errors"]
    metadata = json.loads(str(data["metadata"]))
    return syndromes, errors, metadata


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    config: TrainingConfig,
    syndrome_size: int,
) -> train_state.TrainState:
    """Create initial training state."""
    # Initialize model
    dummy_input = jnp.zeros((1, syndrome_size), dtype=jnp.int32)
    params = model.init(rng, dummy_input, train=False)

    # Create optimizer with warmup schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.num_epochs * 10000,  # Approximate
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


def compute_loss(params, apply_fn, batch, rng, train: bool = True):
    """Compute cross-entropy loss."""
    syndromes, errors = batch

    if train:
        logits = apply_fn(params, syndromes, train=True, rngs={"dropout": rng})
    else:
        logits = apply_fn(params, syndromes, train=False)

    # One-hot encode errors
    labels = jax.nn.one_hot(errors, 4)

    # Cross-entropy loss per qubit, averaged
    loss = optax.softmax_cross_entropy(logits, labels)
    loss = jnp.mean(loss)

    return loss


def compute_accuracy(logits: jnp.ndarray, errors: jnp.ndarray) -> jnp.ndarray:
    """Compute per-qubit accuracy."""
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == errors)
    return accuracy


def compute_logical_accuracy(logits: jnp.ndarray, errors: jnp.ndarray) -> jnp.ndarray:
    """Compute accuracy of recovering the full error pattern."""
    predictions = jnp.argmax(logits, axis=-1)
    # Check if entire error pattern is correct
    sample_correct = jnp.all(predictions == errors, axis=-1)
    return jnp.mean(sample_correct)


@jax.jit
def train_step(state, batch, rng):
    """Single training step."""

    def loss_fn(params):
        return compute_loss(params, state.apply_fn, batch, rng, train=True)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    """Evaluation step."""
    syndromes, errors = batch
    logits = state.apply_fn(state.params, syndromes, train=False)
    loss = compute_loss(state.params, state.apply_fn, batch, None, train=False)
    acc = compute_accuracy(logits, errors)
    logical_acc = compute_logical_accuracy(logits, errors)
    return loss, acc, logical_acc


def train_epoch(state, train_syndromes, train_errors, config, rng):
    """Train for one epoch."""
    num_samples = len(train_syndromes)
    num_batches = num_samples // config.batch_size

    # Shuffle data
    perm = jax.random.permutation(rng, num_samples)
    train_syndromes = train_syndromes[perm]
    train_errors = train_errors[perm]

    epoch_loss = 0.0
    for i in range(num_batches):
        batch_rng = jax.random.fold_in(rng, i)
        start = i * config.batch_size
        end = start + config.batch_size

        batch = (
            jnp.array(train_syndromes[start:end]),
            jnp.array(train_errors[start:end]),
        )

        state, loss = train_step(state, batch, batch_rng)
        epoch_loss += float(loss)

    return state, epoch_loss / num_batches


def evaluate(state, val_syndromes, val_errors, config):
    """Evaluate on validation set."""
    num_samples = len(val_syndromes)
    num_batches = (num_samples + config.batch_size - 1) // config.batch_size

    total_loss = 0.0
    total_acc = 0.0
    total_logical_acc = 0.0

    for i in range(num_batches):
        start = i * config.batch_size
        end = min(start + config.batch_size, num_samples)

        batch = (
            jnp.array(val_syndromes[start:end]),
            jnp.array(val_errors[start:end]),
        )

        loss, acc, logical_acc = eval_step(state, batch)
        total_loss += float(loss) * (end - start)
        total_acc += float(acc) * (end - start)
        total_logical_acc += float(logical_acc) * (end - start)

    return (
        total_loss / num_samples,
        total_acc / num_samples,
        total_logical_acc / num_samples,
    )


def save_model(
    state: train_state.TrainState,
    config: ModelConfig,
    metadata: dict,
    output_path: str,
):
    """Save model to pickle file for C++ inference."""
    # Convert parameters to numpy for portability
    params_np = jax.tree_util.tree_map(np.array, state.params)

    model_data = {
        "params": params_np,
        "config": {
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "mlp_dim": config.mlp_dim,
            "dropout_rate": config.dropout_rate,
            "num_classes": config.num_classes,
        },
        "metadata": metadata,
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Saved model to {output_path}")


def export_to_onnx(
    state: train_state.TrainState,
    model: nn.Module,
    syndrome_size: int,
    output_path: str,
):
    """Export model to ONNX format (optional, requires jax2onnx)."""
    try:
        import jax2onnx
    except ImportError:
        logger.warning("jax2onnx not available, skipping ONNX export")
        return

    # This is a placeholder - actual export would require more setup
    logger.info(f"ONNX export not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Train ML decoder for QEC")
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data (.npz)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data (.npz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="decoder_model.pkl",
        help="Output model path",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="transformer",
        choices=["transformer", "mlp"],
        help="Model architecture",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of Transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    if not JAX_AVAILABLE:
        logger.error("JAX/Flax not available. Install with: pip install jax jaxlib flax optax")
        sys.exit(1)

    # Load training data
    logger.info(f"Loading training data from {args.train_data}")
    train_syndromes, train_errors, train_metadata = load_dataset(args.train_data)
    logger.info(f"Loaded {len(train_syndromes)} training samples")
    logger.info(f"  Syndrome shape: {train_syndromes.shape}")
    logger.info(f"  Errors shape: {train_errors.shape}")

    # Load validation data
    if args.val_data:
        logger.info(f"Loading validation data from {args.val_data}")
        val_syndromes, val_errors, _ = load_dataset(args.val_data)
        logger.info(f"Loaded {len(val_syndromes)} validation samples")
    else:
        # Split training data
        n_val = int(len(train_syndromes) * 0.1)
        val_syndromes, val_errors = train_syndromes[-n_val:], train_errors[-n_val:]
        train_syndromes, train_errors = train_syndromes[:-n_val], train_errors[:-n_val]
        logger.info(f"Split: {len(train_syndromes)} train, {len(val_syndromes)} val")

    # Get dimensions
    syndrome_size = train_syndromes.shape[-1]
    num_qubits = train_errors.shape[-1]

    logger.info(f"Syndrome size: {syndrome_size}, Num qubits: {num_qubits}")

    # Create model
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    if args.model_type == "transformer":
        model = TransformerDecoder(
            config=model_config,
            num_qubits=num_qubits,
            syndrome_size=syndrome_size,
        )
    else:
        model = MLPDecoder(
            hidden_dims=[args.hidden_dim] * args.num_layers,
            num_qubits=num_qubits,
        )

    # Create training config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
    )

    # Initialize training
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, model, training_config, syndrome_size)
    logger.info(f"Model initialized with {sum(p.size for p in jax.tree_util.tree_leaves(state.params)):,} parameters")

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        rng, epoch_rng = jax.random.split(rng)

        # Train
        state, train_loss = train_epoch(
            state, train_syndromes, train_errors, training_config, epoch_rng
        )

        # Evaluate
        val_loss, val_acc, val_logical_acc = evaluate(
            state, val_syndromes, val_errors, training_config
        )

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Logical Acc: {val_logical_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = state
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= training_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Save best model
    if best_state is not None:
        state = best_state

    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

    # Final evaluation
    val_loss, val_acc, val_logical_acc = evaluate(
        state, val_syndromes, val_errors, training_config
    )
    logger.info(f"Final metrics:")
    logger.info(f"  Validation Loss: {val_loss:.4f}")
    logger.info(f"  Per-qubit Accuracy: {val_acc:.4f}")
    logger.info(f"  Logical Accuracy: {val_logical_acc:.4f}")

    # Save model
    metadata = {
        **train_metadata,
        "model_type": args.model_type,
        "syndrome_size": syndrome_size,
        "num_qubits": num_qubits,
        "best_val_acc": float(best_val_acc),
        "final_val_logical_acc": float(val_logical_acc),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_model(state, model_config, metadata, args.output)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
