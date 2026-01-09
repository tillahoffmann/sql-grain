import shutil
import sqlite3
from pathlib import Path

import pytest

from sqlgrain import Sqlite3DataSource


def test_data_source_length(test_db: Path) -> None:
    """Data source reports correct number of keys."""
    source = Sqlite3DataSource(
        test_db,
        key_query="SELECT id AS user_id FROM users",
        record_query="SELECT item, amount FROM orders WHERE user_id = :user_id",
    )
    assert len(source) == 2


def test_data_source_getitem(test_db: Path) -> None:
    """Data source returns records as column-oriented dicts."""
    source = Sqlite3DataSource(
        test_db,
        key_query="SELECT id AS user_id FROM users ORDER BY id",
        record_query="SELECT item, amount FROM orders WHERE user_id = :user_id ORDER BY item",
    )

    # First user (Alice) has 2 orders
    record = source[0]
    assert record["item"] == ("apple", "banana")
    assert record["amount"] == (10, 20)

    # Second user (Bob) has 1 order
    record = source[1]
    assert record["item"] == ("cherry", "orange")
    assert record["amount"] == (30, 40)


def test_data_source_with_record_params(test_db: Path) -> None:
    """Data source supports additional record parameters."""
    source = Sqlite3DataSource(
        test_db,
        key_query="SELECT id AS user_id FROM users ORDER BY id",
        record_query="SELECT item FROM orders WHERE user_id = :user_id AND amount >= :min_amount",
        record_params={"min_amount": 15},
    )

    # Alice: only banana (amount=20) meets threshold
    record = source[0]
    assert record["item"] == ("banana",)

    # Bob: cherry (amount=30) and orange (amount=40) meets threshold
    record = source[1]
    assert record["item"] == ("cherry", "orange")


def test_data_source_closing(test_db: Path) -> None:
    with Sqlite3DataSource(
        test_db, "SELECT id AS user_id FROM users ORDER BY id", "<EMPTY>"
    ) as data_source:
        pass
    assert not hasattr(data_source._local, "conn")

    with Sqlite3DataSource(
        test_db, "SELECT id AS user_id FROM users ORDER BY id", "<EMPTY>"
    ) as data_source:
        # Access keys to execute a query.
        data_source.keys
    assert hasattr(data_source._local, "conn")

    # Check the database is closed.
    with pytest.raises(sqlite3.ProgrammingError):
        data_source._get_conn().execute("SELECT id FROM users")


def test_data_source_repr(test_db: Path, tmp_path: Path) -> None:
    kwargs = {
        "database": test_db,
        "key_query": "key query",
        "record_query": "record query",
        "key_params": {"key": "param"},
        "record_params": {"record1": "param1", "record2": "param2"},
    }
    ref = Sqlite3DataSource(**kwargs)
    ref_repr = repr(ref)

    # Reordering dict entries leaves the fingerprint unchanged.
    assert ref_repr == repr(
        Sqlite3DataSource(
            **(kwargs | {"record_params": {"record2": "param2", "record1": "param1"}})
        )
    )

    # Make sure things are different when we change arguments.
    other_db = tmp_path / "other.db"
    shutil.copy(test_db, other_db)

    others = [
        {"database": other_db},
        {"key_query": "other"},
        {"record_query": "other"},
        {"key_params": {"something": "else"}},
        {"record_params": {"something": "different"}},
    ]

    for other in others:
        assert ref_repr != repr(Sqlite3DataSource(**(kwargs | other)))

    # Check modification.
    with open(test_db, "w") as fp:
        fp.write("different file")

    assert ref_repr != repr(ref)
