from pathlib import Path
from typing import Iterable, Sequence, cast

import numpy as np
import pytest
from grain import MapDataset

from sqlgrain import Sqlite3DataSource
from sqlgrain.records import decode_record, from_array_record, to_array_record


@pytest.mark.parametrize("use_dataset", [False, True])
@pytest.mark.parametrize("compression", [None, "zstd"])
def test_to_array_record(
    test_db: Path, tmp_path: Path, use_dataset: bool, compression: str | None
) -> None:
    """Test round-trip serialization to ArrayRecord format."""
    sql_source = Sqlite3DataSource(
        test_db,
        key_query="SELECT id AS user_id FROM users",
        record_query="SELECT item, amount FROM orders WHERE user_id = :user_id",
    )
    if use_dataset:
        records = (
            MapDataset.source(sql_source)
            .map(
                lambda record: {key: np.asarray(value) for key, value in record.items()}
            )
            .batch(2)
        )
    else:
        records = sql_source

    path = tmp_path / "records"
    to_array_record(
        cast(Iterable, records), path, aux={"hello": "world"}, compression=compression
    )

    # Check just the source.
    ar_source, metadata = from_array_record(path)
    assert metadata["aux"]["hello"] == "world"
    for data in ar_source:
        assert isinstance(data, bytes)

    # Check decoding.
    ar_dataset = MapDataset.source(cast(Sequence, ar_source)).map(decode_record)
    assert len(ar_dataset) == len(records)
    for a, b in zip(cast(Iterable, records), ar_dataset):
        for key in a:
            if isinstance(a[key], list):
                assert list(a[key]) == b[key]
            else:
                np.testing.assert_array_equal(a[key], b[key])


def test_shard_every(tmp_path: Path) -> None:
    """Sharding by record count creates expected number of shards."""
    records = [{"x": i} for i in range(10)]
    path = tmp_path / "sharded"
    to_array_record(records, path, shard_every=3)

    _, metadata = from_array_record(path)
    # 10 records with shard_every=3 should create 4 shards (0,3,6,9 start new shards)
    assert len(metadata["shards"]) == 4
    assert metadata["num_records"] == 10


def test_shard_size(tmp_path: Path) -> None:
    """Sharding by byte size creates multiple shards."""
    # Create records with known size (~445 bytes serialized each)
    records = [{"data": np.zeros(100, dtype=np.float32)} for _ in range(5)]
    path = tmp_path / "sharded"
    # With shard_size=500 and ~445 byte records, each shard holds 1 record
    # (445 < 500, but 445 + 445 > 500 triggers new shard)
    with pytest.warns(UserWarning, match="shard_size=500 is less than 1MB"):
        to_array_record(records, path, shard_size=500)

    ar_source, metadata = from_array_record(path)
    # 5 records at ~445 bytes each with 500 byte limit = 5 shards
    assert len(metadata["shards"]) == 5
    assert metadata["num_records"] == 5

    # Verify all records can be decoded
    for data in ar_source:
        record = decode_record(data)
        np.testing.assert_array_equal(record["data"], np.zeros(100, dtype=np.float32))
