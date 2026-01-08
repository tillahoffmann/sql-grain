# MovieLens

This example illustrates training a transformer recommendation model on the stable [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/latest/). The dataset comprises 1,000,209 ratings for 3,883 movies made by 6,040 users. There are at least 20 ratings per user.

## Running the Example

You can run the example end-to-end from the command line by following the setup instructions in the [main README.md](../../README.md) and running `uv run make` from this directory.

## Data Loading Performance

You can test the performance of the data loader using a "dry run".

```bash
$ uv run python train.py --dry-run data/ml-1m.db
```

To speed up the data loader, we can first convert the dataset to an ArrayRecord dataset using the `--cache` option. This will convert the dataset on the first invocation and then use the cached dataset for subsequent runs.

```bash
$ uv run python train.py --dry-run --cache data/ml-1m.db
```

## Stack

- `sqlgrain` to construct "sentences" for the transformer.
- `flax` for building the transformer.
- `optax` for optimization.
- `orbax` for checkpointing.
