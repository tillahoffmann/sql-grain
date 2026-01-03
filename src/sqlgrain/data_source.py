import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Mapping, Self, Sequence, SupportsIndex, cast

from grain.sources import RandomAccessDataSource


class _Local(threading.local):
    """Type annotation for thread-local connections."""

    conn: sqlite3.Connection


def _get_column_names(cursor: sqlite3.Cursor) -> list[str]:
    """Get the names of columns from a sqlite3 database cursor."""
    return [column for (column, *_) in cursor.description]


class Sqlite3DataSource(RandomAccessDataSource[Mapping[str, Sequence]]):
    """Data source to fetch records from a :mod:`sqlite3` database.

    Records are obtained in two steps. First, keys to uniquely identify records in the
    database are fetched using the :attr:`key_query`. Second, :meth:`__getitem__`
    returns a record given an integer index used to select an element in the list of
    keys. Each record is a dictionary mapping column names to lists of values.

    .. note::

        For parallel data loading, use :meth:`~grain.IterDataset.mp_prefetch` with
        multiple worker processes rather than multiple threads. The :mod:`sqlite3`
        module holds the Python GIL during query result fetching, causing thread-based
        parallelism to degrade. Multiprocessing avoids this by giving each worker its
        own GIL.

    Args:
        database: Path to database.
        key_query: Query to fetch keys from the database which identify records to be
            returned by the data source. This query is executed once, and it *must* be
            deterministic.
        record_query: Query to select records from the database given a key which is
            passed to the query as named parameters. The results of the query are
            returned as a dictionary keyed by column name.
        key_params: Query parameters passed to the key query.
        record_params: Additional query parameters passed to the record query. Parameter
            names must not overlap with the values returned by the :attr:`key_query`.
    """

    def __init__(
        self,
        database: Path | str,
        key_query: str,
        record_query: str,
        key_params: dict | None = None,
        record_params: dict | None = None,
    ) -> None:
        self.database = Path(database)
        self.key_query = key_query
        self.record_query = record_query
        self.key_params = key_params or {}
        self.record_params = record_params or {}

        self._keys: list[dict[str, Any]] | None = None
        self._local = cast(_Local, threading.local())

    def _get_conn(self) -> sqlite3.Connection:
        # We use `hasattr` rather than using an `Optional[sqlite3.Connection]` because
        # unpickling may happen in a different thread than accessing the connection. If
        # we only create the thread-local object in the `__setstate__` method, accessing
        # the optional attribute would fail in a different thread.
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                f"file:{self.database}?mode=ro", uri=True
            )
        return self._local.conn

    @property
    def keys(self) -> list[dict]:
        """Unique keys of the data source."""
        if self._keys is None:
            conn = self._get_conn()
            cursor = conn.execute(self.key_query, self.key_params)
            columns = _get_column_names(cursor)
            conflict = set(columns) & set(self.record_params)
            if conflict:
                raise ValueError(
                    "Conflicting parameter names from key query and explicit "
                    f"parameters for record query: {conflict}."
                )
            self._keys = [dict(zip(columns, row)) for row in cursor]
        return self._keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: SupportsIndex) -> Mapping[str, Sequence]:
        # Execute the query and get column names if required.
        params = self.record_params | self.keys[index]
        conn = self._get_conn()
        cursor = conn.execute(self.record_query, params)

        # Transpose rows to columns. zip(*rows) produces tuples in C, avoiding
        # Python-level list construction overhead.
        columns = _get_column_names(cursor)
        rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"Failed to fetch data for record with key '{index}'.")
        return dict(zip(columns, zip(*rows)))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn:
            conn.close()

    def __repr__(self) -> str:
        # We need a `__repr__` implementation because `grain` uses it to assess if a
        # data source is equivalent to another. This is a fuzzy hash implementation.
        # I.e., the same hash does not guarantee the same database. But different hashes
        # guarantee different databases.

        key_query_hash = hashlib.sha256(self.key_query.encode()).hexdigest()
        record_query_hash = hashlib.sha256(self.record_query.encode()).hexdigest()
        last_modified = self.database.stat().st_mtime
        return (
            f"<{self.__class__.__name__} with database path '{self.database}' "
            f"(last modified at {last_modified}), key query hash "
            f"'{key_query_hash}', record query hash '{record_query_hash}'>"
        )

    def __setstate__(self, values: dict) -> None:
        self.__dict__.update(values)
        # Create attributes that were not included in the state.
        self._keys = None
        self._local = cast(_Local, threading.local())

    def __getstate__(self) -> dict:
        # Required for pickling datasets across multiple workers.
        return {
            "database": self.database,
            "key_query": self.key_query,
            "record_query": self.record_query,
            "key_params": self.key_params,
            "record_params": self.record_params,
        }
