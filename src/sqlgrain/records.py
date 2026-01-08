import json
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import msgpack
from array_record.python.array_record_module import ArrayRecordWriter
from grain.sources import ArrayRecordDataSource


def _default_encode(obj: Any) -> Any:
    """Default encoder, supporting numpy types."""
    # We use a "soft" check so we don't have to depend on NumPy or JAX.
    type_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
    if type_name in {"numpy.ndarray", "jaxlib._jax.ArrayImpl"}:
        import numpy as np

        arr = np.asarray(obj)
        return {
            "__nd__": True,
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "data": arr.tobytes(),
        }
    raise TypeError(f"Unknown type: {type(obj)}")


def _default_decode_hook(obj: dict) -> Any:
    """Default decoder, supporting numpy types."""
    if obj.get("__nd__"):
        import numpy as np

        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


def decode_record(data: bytes, decode_hook: Callable[[dict], Any] | None = None) -> Any:
    """Decode a msgpack record. This is a thin wrapper around :func:`msgpack.loads`
    which supports decoding of NumPy arrays.

    Args:
        data: Message to decode.
        decode_hook: Hook for decoding custom objects.

    Returns:
        Decoded record.
    """
    decode_hook = decode_hook or _default_decode_hook
    return msgpack.loads(data, object_hook=decode_hook)


def to_array_record(
    records: Iterable,
    path: Path | str,
    compression: str | None = None,
    shard_every: int | None = None,
    shard_size: int | None = None,
    key: str | None = None,
    encode: Callable[[Any], Any] | None = None,
    aux: Any = None,
) -> None:
    """Convert an iterable of records to Array Record files with pattern
    :code:`data-#####.arrayrecord` and a metadata file :code:`_metadata.json`. Records
    are serialized using msgpack.

    Args:
        records: Records to save.
        path: Output directory.
        compression: Compression setting passed to ArrayRecordWriter (e.g., "zstd").
        shard_every: Shard every N records.
        shard_size: Shard every N bytes.
        key: Key uniquely identifying the records for consistency checks.
        encode: Encoder function to transform non-standard types for msgpack.
        aux: JSON-serializable auxiliary information to add to the metadata file.
    """
    path = Path(path)
    path.mkdir(parents=True)
    compression = compression or ""
    encode = encode or _default_encode

    if shard_size is not None and shard_size < 1_000_000:
        warnings.warn(
            f"shard_size={shard_size} is less than 1MB. ArrayRecord files have a "
            "minimum size of 128KB, so small shards may waste significant disk space.",
            stacklevel=2,
        )

    writer = None
    size = 0
    shard_paths = []

    i = -1
    try:
        for i, record in enumerate(records):
            data = msgpack.packb(record, default=encode)
            if not data:
                raise ValueError(f"Failed to serialize record at index {i}.")

            if (
                writer is None
                or (shard_every and i % shard_every == 0)
                or (shard_size and shard_size < size + len(data))
            ):
                if writer:
                    writer.close()
                    writer = None
                name = f"data-{len(shard_paths):05}.arrayrecord"
                writer = ArrayRecordWriter(str(path / name), compression)
                shard_paths.append(name)
                size = 0

            writer.write(data)
            size += len(data)
    finally:
        if writer:
            writer.close()

    with open(path / "_metadata.json", "w") as fp:
        json.dump(
            {
                "key": key,
                "num_records": i + 1,
                "shards": shard_paths,
                "aux": aux,
            },
            fp,
        )


def from_array_record(
    path: Path | str, key: str | None = None
) -> tuple[ArrayRecordDataSource, dict]:
    """Load (sharded) array record files and metadata from a directory.

    Args:
        path: Path to load from.
        key: Key uniquely identifying the records for consistency checks.

    Returns:
        Tuple of data source and metadata.
    """
    path = Path(path)
    with open(path / "_metadata.json") as fp:
        metadata = json.load(fp)

    assert key is None or metadata["key"] == key, (
        f"Key '{metadata['key']}' in '{path}' does not match '{key}'."
    )

    paths = [path / shard for shard in metadata["shards"]]
    return (ArrayRecordDataSource(paths), metadata)
