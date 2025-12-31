from flax import nnx
from jax import numpy as jnp


class TransformerBlock(nnx.Module):
    """Single transformer block with masked self-attention, two-layer dense
    transformation, residual connections, and pre-layer norm.

    Args:
        num_features: Number of embedding dimensions.
        rngs: Random number generator sequence.
    """

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        dropout_rate: float,
        num_hidden: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm1 = nnx.LayerNorm(num_features=num_features, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=num_features,
            use_bias=False,
            rngs=rngs,
            dropout_rate=dropout_rate,
            decode=False,
            deterministic=False,
        )
        self.norm2 = nnx.LayerNorm(num_features=num_features, rngs=rngs)
        self.feed_forward = nnx.Sequential(
            nnx.Linear(in_features=num_features, out_features=num_hidden, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(in_features=num_hidden, out_features=num_features, rngs=rngs),
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Construct a causal attention mask.
        batch_size, num_tokens, num_features = x.shape
        mask = jnp.tril(jnp.ones((num_tokens, num_tokens)))

        # Self-attention layer.
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x, mask=mask)
        # No dropout here, because the attention layer already has dropout.
        x = x + shortcut

        # Dense feed-forward layer.
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


class TransformerModel(nnx.Module):
    """Transformer model to predict users interaction with items, following
    https://arxiv.org/abs/1808.09781.

    Args:
        context_size: Maximum number of movies in a user context. Longer rating
            histories are truncated.
        num_features: Number of embedding dimensions.
        num_layers: Number of transformer layers.
        num_movies: Number of unique movies.
        dropout: Dropout rate.
        rngs: Random number generator sequence.
    """

    def __init__(
        self,
        *,
        context_size: int,
        num_features: int,
        num_layers: int,
        num_movies: int,
        dropout_rate: float,
        num_heads: int,
        num_hidden: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.context_size = context_size
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_movies = num_movies
        self.dropout_rate = dropout_rate

        self.movie_embedding = nnx.Embed(num_movies, num_features, rngs=rngs)
        self.pos_embedding = nnx.Embed(context_size, num_features, rngs=rngs)
        self.blocks = nnx.Sequential(
            *(
                TransformerBlock(
                    num_features,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                    num_heads=num_heads,
                    num_hidden=num_hidden,
                )
                for _ in range(self.num_layers)
            )
        )
        self.initial_dropout = nnx.Dropout(self.dropout_rate, rngs=rngs)
        self.final_norm = nnx.LayerNorm(num_features=num_features, rngs=rngs)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        pos = jnp.arange(inputs.shape[-1])
        embedding = self.movie_embedding(inputs) + self.pos_embedding(pos)
        embedding = self.initial_dropout(embedding)
        embedding = self.blocks(embedding)
        embedding = self.final_norm(embedding)
        return embedding
