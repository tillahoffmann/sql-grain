import argparse
from pathlib import Path
from typing import cast

import grain
import numpy as np
from absl import flags
from flax import nnx
from grain._src.python.dataset import stats
from jax import numpy as jnp
from tqdm import tqdm

from sqlgrain import Sqlite3DataSource
from sqlgrain.util import assert_true, encode_many


class TransformerBlock(nnx.Module):
    """Single transformer block with masked self-attention, two-layer dense
    transformation, residual connections, and pre-layer norm.

    Args:
        num_features: Number of embedding dimensions.
        rngs: Random number generator sequence.
    """

    def __init__(self, num_features: int, rngs: nnx.Rngs) -> None:
        self.num_features = num_features


class TransformerModel(nnx.Module):
    """Transformer model to predict users interaction with items, following
    https://arxiv.org/abs/1808.09781.

    Args:
        context_size: Maximum number of movies in a user context. Longer rating
            histories are truncated.
        num_features: Number of embedding dimensions.
        num_layers: Number of transformer layers.
        num_movies: Number of unique movies.
        rngs: Random number generator sequence.
    """

    def __init__(
        self,
        *,
        context_size: int,
        num_features: int,
        num_layers: int,
        num_movies: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.context_size = context_size
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_movies = num_movies

        self.movie_embedding = nnx.Embed(num_movies, num_features, rngs=rngs)
        self.pos_embedding = nnx.Embed(context_size, num_features, rngs=rngs)
        self.blocks = nnx.Sequential(
            *(TransformerBlock(num_features, rngs=rngs) for _ in range(self.num_layers))
        )

    def __call__(self, inputs: jnp.ndarray, outputs: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(1)


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
    return batch[:, :-1], batch[:, 1:]


def create_dataset(
    data_source: Sqlite3DataSource,
    *,
    train: bool,
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
        train: Whether this is the training set.
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
    if train:
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
        .map(encode_many(tokenizer, frozen=not train, default=tokenizer["<UNK>"]))
        # Mini-batches for optimization. We drop incomplete batches during training to
        # prevent noisy gradients for small batches.
        .batch(
            batch_size,
            drop_remainder=train,
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


class _Args:
    database: Path
    context_size: int
    batch_size: int
    dry_run: bool
    num_epochs: int


def __main__() -> None:
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
        "--dry-run",
        action="store_true",
        help="Perform a dry-run over the dataset without optimization.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of optimization epochs to run.",
        default=201,
    )
    parser.add_argument("database", type=Path, help="Path to MovieLens database.")
    args = cast(_Args, parser.parse_args())

    # Enable profiling if we're only doing a dry run. We need to mark absl flags
    # as parsed since we use argparse instead of absl's app.run().
    if args.dry_run:
        grain.config.update("py_debug_mode", True)
        flags.FLAGS.mark_as_parsed()

    # Create data sources, dropping the last 2 items for training, last 1 item for
    # validation, and reserving the last item for later testing. We then turn the data
    # *source* into a data*set* for iteration.
    data_sources = {
        "train": create_data_source(args.database, drop_last_k=2),
        "valid": create_data_source(args.database, drop_last_k=1),
    }
    tokenizer = {"<START>": 0, "<PAD>": 1, "<UNK>": 2}
    seeds = (12345, 67890)
    datasets = {
        split: create_dataset(
            data_source,
            train=split == "train",
            seed=seed,
            tokenizer=tokenizer,
            context_size=args.context_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
        )
        for seed, (split, data_source) in zip(seeds, data_sources.items())
    }

    # We restrict to one thread because concurrency leads to database lock competition
    # and actually slows things down. This requires we explicitly call `to_iter_dataset`
    # because default iteration uses 16 threads.
    num_steps = len(datasets["train"])
    train_iter_dataset = datasets["train"].to_iter_dataset(
        grain.ReadOptions(num_threads=1)
    )
    train_iter = iter(train_iter_dataset)
    for inputs, outputs in tqdm(
        train_iter,
        total=num_steps,
    ):
        assert inputs.shape == (args.batch_size, args.context_size)
        assert outputs.shape == (args.batch_size, args.context_size)

    if args.dry_run:
        summary = train_iter._stats._get_execution_summary()  # type: ignore[union-attr]
        print(stats.pretty_format_summary(summary))

    # breakpoint()


if __name__ == "__main__":
    __main__()
