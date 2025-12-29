from typing import Callable, MutableMapping, TypeVar

T = TypeVar("T")


def encode(
    mapping: MutableMapping[T, int], *, frozen: bool = True, default: int | None = None
) -> Callable[[T], int]:
    """Encode a single token to an integer.

    Args:
        mapping: Mapping from tokens to integers which is mutated in-place.
        frozen: If :code:`True`, freeze the mapping so unknown tokens raise an exception
            or use the :code:`default` value if provided. If :code:`False`, update the
            mapping with the next consecutive integer when encountering an unknown
            token.
        default: Default value to use if :code:`frozen` is :code:`True` and the token is
            unknown.

    Returns:
        Encoder that maps a token to an integer.
    """

    def _encoder(token: T) -> int:
        try:
            return mapping[token]
        except KeyError:
            if frozen:
                if default is not None:
                    return default
                raise
            new = len(mapping)
            mapping[token] = new
            return new

    return _encoder


def encode_many(
    mapping: MutableMapping[T, int], *, frozen: bool = True, default: int | None = None
) -> Callable[[list[T]], list[int]]:
    """Encode a list of tokens to integers.

    Args:
        mapping: Mapping from tokens to integers which is mutated in-place.
        frozen: If :code:`True`, freeze the mapping so unknown tokens raise an exception
            or use the :code:`default` value if provided. If :code:`False`, update the
            mapping with the next consecutive integer when encountering an unknown
            token.
        default: Default value to use if :code:`frozen` is :code:`True` and the token is
            unknown.

    Returns:
        Encoder that maps a list of tokens to a list of integers.
    """
    _encode_one = encode(mapping, frozen=frozen, default=default)

    def _encoder(tokens: list[T]) -> list[int]:
        return [_encode_one(t) for t in tokens]

    return _encoder


def assert_true(predicate: Callable[[T], bool], *args, **kwargs) -> Callable[[T], T]:
    """Assert that a predicate is true for each record.

    Args:
        predicate: Predicate to evaluate on each record.
        *args: Positional arguments passed to :code:`predicate`.
        **kwargs: Keyword arguments passed to :code:`predicate`.

    Returns:
        Function that verifies the predicate is satisfies and passes through the input.
    """

    def _wrapper(input: T) -> T:
        if not predicate(input, *args, **kwargs):
            raise ValueError("Record does not satisfy the predicate.")
        return input

    return _wrapper
