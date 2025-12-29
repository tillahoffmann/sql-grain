import argparse
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import grain
import numpy as np
import optax
from absl import flags
from flax import nnx
from grain._src.python.dataset import stats
from jax import numpy as jnp
from model import TransformerModel
from tqdm import tqdm

from sqlgrain import Sqlite3DataSource
from sqlgrain.util import assert_true, encode_many


def create_data_source(database: Path, drop_last_k: int) -> Sqlite3DataSource:
    """Create a data source that groups movies by user but drops the most recent
    :code:`k` ratings.

    Args:
        database: Path to MovieLens database.
        drop_last_k: How many of the most recent ratings to drop.

    Returns:
        Data source.
    """
    # The `ROW_NUMBER() OVER (ORDER BY "timestamp" DESC) AS "rank"` creates a rank of
    # how recent each rating is, with the most recent having rank 1 and older items
    # having higher rank. We drop the most recent `k` items by filtering on the rank and
    # then sort by the rank in DESC order to restore the chronological order.

    return Sqlite3DataSource(
        database,
        key_query="SELECT id AS user_id FROM users",
        record_query="""
        WITH ordered_ratings AS (
            SELECT
                "movie_id",
                ROW_NUMBER() OVER (ORDER BY "timestamp" DESC) AS "rank"
            FROM ratings
            WHERE user_id = :user_id
        )
        SELECT movie_id
        FROM ordered_ratings
        WHERE "rank" > :drop_last_k
        ORDER BY "rank" DESC
        """,
        record_params={"drop_last_k": drop_last_k},
    )


def make_inputs_outputs(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create shifted input-output pairs from a batched array."""
    return batch[..., :-1], batch[..., 1:]


def create_dataset(
    data_source: Sqlite3DataSource,
    *,
    seed: int | None,
    tokenizer: dict,
    context_size: int,
    batch_size: int,
    num_epochs: int,
) -> grain.MapDataset:
    """Create a dataset which transforms the raw SQLite results to a form amenable to
    Flax NNX.

    Args:
        data_source: Data source to load from.
        seed: Random number generator seed.
        tokenizer: Mapping from movie ids to consecutive integers.
        context_size: Size of the context window.
        num_epochs: Number of optimization epochs.

    Returns:
        Dataset that can be iterated over.
    """
    dataset = grain.MapDataset.source(data_source)
    if seed is not None:
        dataset = dataset.seed(seed)
    dataset = dataset.shuffle().repeat(num_epochs)
    dataset = (
        dataset.map(lambda record: record["movie_id"])
        # Prepend <START> token and pad to the context size.
        .map(
            lambda movie_ids: (
                ["<START>", *movie_ids] + ["<PAD>"] * (context_size - len(movie_ids))
            )[: context_size + 1]
        )
        # Encode to consecutive integers.
        .map(encode_many(tokenizer, frozen=False, default=tokenizer["<UNK>"]))
        # Mini-batches for optimization. We drop incomplete batches during training to
        # prevent noisy gradients for small batches.
        .batch(
            batch_size,
            drop_remainder=True,
            batch_fn=lambda x: np.asarray(x, dtype=np.int32),
        )
        # Turn into shifted input-output pairs for self-supervised optimization.
        .map(make_inputs_outputs)
        # Ensure we have the right shapes.
        .map(
            assert_true(
                lambda x: x[0].shape[1] == context_size
                and x[1].shape[1] == context_size
            )
        )
    )
    return dataset


def evaluate_end_loss_mask(labels: jnp.ndarray, end_token: int) -> jnp.ndarray:
    """Evaluate a mask such that the loss from elements *after* the first END token do
    not contribute to the loss."""
    # Mask out anything after the first end of playlist token.
    is_eop = labels == end_token
    has_eop = is_eop.any(axis=1)
    length = labels.shape[-1]
    first = jnp.where(has_eop, jnp.argmax(is_eop, axis=1), length - 1)
    return jnp.arange(length) <= first[:, None]


def loss_fn(
    model: TransformerModel, inputs: jnp.ndarray, outputs: jnp.ndarray, end_token: int
) -> jnp.ndarray:
    """Softmax cross-entropy loss for all tokens except trailing padding beyond the
    end-of-context marker (which also happens to be a padding marker).

    Args:
        model: Transformer model to make predictions.
        inputs: Input tokens with shape (batch_size, context_size).
        outputs: Target tokens matching the inputs.
        end_token: Token indicating the end of the context.

    Returns:
        Scalar loss.
    """
    batch_size, context_size = inputs.shape
    transformed_embedding = model(inputs)
    assert transformed_embedding.shape == (batch_size, context_size, model.num_features)

    logits = jnp.vecdot(
        transformed_embedding.reshape(
            (batch_size * context_size, 1, model.num_features)
        ),
        model.movie_embedding.embedding[...],
    )
    elementwise_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, outputs.reshape((batch_size * context_size,))
    )
    mask = evaluate_end_loss_mask(outputs, end_token)
    return cast(jnp.ndarray, elementwise_loss @ mask.ravel() / mask.sum())


@nnx.jit
def train_step(
    model: TransformerModel,
    optimizer: nnx.Optimizer,
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    end_token: int,
) -> jnp.ndarray:
    """Single JIT-compiled training step."""
    loss, grads = nnx.value_and_grad(loss_fn)(model, inputs, outputs, end_token)
    optimizer.update(model, grads)
    return loss


class _Args:
    database: Path
    context_size: int
    batch_size: int
    dry_run: bool
    num_epochs: int
    num_features: int
    num_layers: int
    learning_rate: float
    dropout: float
    num_heads: int


def main() -> None:
    """Main training loop based on
    https://github.com/kang205/SASRec/blob/master/main.py.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-size",
        type=int,
        help="Maximum number of movies in a user context. Longer histories are truncated.",
        default=50,  # They use 200 for MovieLens, but we restrict to 50 for demo.
    )
    parser.add_argument(
        "--num-features", type=int, help="Number of embedding dimensions", default=50
    )
    parser.add_argument(
        "--batch-size", type=int, help="Mini-batch size for optimization.", default=128
    )
    parser.add_argument(
        "--num-layers", type=int, help="Number of transformer layers.", default=2
    )
    # We use a lower dropout rate here than in the reference script (0.5).
    parser.add_argument(
        "--dropout", type=float, help="Dropout probability.", default=0.2
    )
    parser.add_argument(
        "--num-heads", help="Number of attention heads.", type=int, default=1
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for Adam optimizer.",
        default=0.001,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry-run over the dataset without optimization.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of optimization epochs to run.",
        default=3 if "CI" in os.environ else 201,
    )
    parser.add_argument("database", type=Path, help="Path to MovieLens database.")
    args = cast(_Args, parser.parse_args())

    # Enable profiling if we're only doing a dry run. We need to mark absl flags
    # as parsed since we use argparse instead of absl's app.run().
    if args.dry_run:
        grain.config.update("py_debug_mode", True)
        flags.FLAGS.mark_as_parsed()

    # Create data source, dropping the last 2 items per user for training (reserving them
    # for later validation/testing). We then turn the data *source* into a data*set* for
    # iteration.
    train_data_source = create_data_source(args.database, drop_last_k=2)
    tokenizer = {"<START>": 0, "<PAD>": 1, "<UNK>": 2}
    train_dataset = create_dataset(
        train_data_source,
        seed=12345,
        tokenizer=tokenizer,
        context_size=args.context_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    # Count distinct movies in the training set (replicating drop_last_k logic).
    with closing(sqlite3.connect(args.database)) as conn:
        (num_movies,) = conn.execute(
            """
            WITH ordered_ratings AS (
                SELECT
                    movie_id,
                    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY "timestamp" DESC) AS "rank"
                FROM ratings
            )
            SELECT COUNT(DISTINCT movie_id)
            FROM ordered_ratings
            WHERE "rank" > 2
            """
        ).fetchone()
    # Account for <START>, <PAD>, <UNK>.
    num_movies += len(tokenizer)

    # Set up the model and optimizer.
    rngs = nnx.Rngs(42)
    model = TransformerModel(
        context_size=args.context_size,
        num_features=args.num_features,
        num_hidden=args.num_features,
        num_layers=args.num_layers,
        rngs=rngs,
        num_movies=num_movies,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
    )
    optimizer = nnx.Optimizer(model, optax.adam(args.learning_rate), wrt=nnx.Param)

    # We restrict to one thread because concurrency leads to database lock competition
    # and actually slows things down. This requires we explicitly call `to_iter_dataset`
    # because default iteration uses 16 threads.
    num_steps = len(train_dataset)
    train_iter_dataset = train_dataset.to_iter_dataset(grain.ReadOptions(num_threads=1))
    train_iter = iter(train_iter_dataset)
    with tqdm(total=num_steps) as progress, closing(train_iter):
        for inputs, outputs in train_iter:
            assert inputs.shape == (args.batch_size, args.context_size)
            assert outputs.shape == (args.batch_size, args.context_size)

            if args.dry_run:
                loss = float("nan")
            else:
                loss = train_step(
                    model, optimizer, inputs, outputs, end_token=tokenizer["<PAD>"]
                )

            progress.update()
            progress.set_description(f"loss={loss:.3f}")

    if args.dry_run:
        summary = train_iter._stats._get_execution_summary()  # type: ignore[union-attr]
        print(stats.pretty_format_summary(summary))


if __name__ == "__main__":
    main()
