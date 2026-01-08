import sqlite3
from pathlib import Path

import pytest


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
        [(1, "apple", 10), (1, "banana", 20), (2, "cherry", 30), (2, "orange", 40)],
    )
    conn.commit()
    conn.close()
    return db_path
