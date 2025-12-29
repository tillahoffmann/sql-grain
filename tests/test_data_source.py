import sqlite3
from pathlib import Path

import pytest

from sqlgrain import Sqlite3DataSource


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Create a temporary database with test data."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE orders (user_id INTEGER, item TEXT, amount INTEGER)")
    conn.executemany("INSERT INTO users VALUES (?, ?)", [(1, "Alice"), (2, "Bob")])
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)",
        [(1, "apple", 10), (1, "banana", 20), (2, "cherry", 30)],
    )
    conn.commit()
    conn.close()
    return db_path


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
    assert record["item"] == ("cherry",)
    assert record["amount"] == (30,)


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

    # Bob: cherry (amount=30) meets threshold
    record = source[1]
    assert record["item"] == ("cherry",)
