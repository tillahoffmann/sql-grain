# 🌾 SQL Grain

SQLite databases as [Grain](https://github.com/google/grain) data sources.

sql-grain lets you prototype ML data pipelines using SQL queries before committing to a production data format. Define your training examples with expressive SQL—joins, window functions, filtering—and iterate quickly without preprocessing. When you're ready to scale, convert to ArrayRecord or similar formats; sql-grain is not designed for large-scale training.

```python
from sqlgrain import Sqlite3DataSource
import grain

source = Sqlite3DataSource(
    "data.db",
    key_query="SELECT id FROM users",
    record_query="SELECT item FROM purchases WHERE user_id = :id ORDER BY timestamp",
)
dataset = grain.MapDataset.source(source).shuffle().batch(32)
```

## Converting to ArrayRecord

Once you're ready to run larger experiments, convert the dataset to ArrayRecords using the `to_array_record` function which serializes records in [msgpack](https://msgpack.org) format with native support for NumPy arrays.

```python
from sqlgrain import to_array_record

to_array_record(source, "output/", shard_every=1000)
```

Grain's [`ArrayRecordDataSource`](https://google-grain.readthedocs.io/en/latest/data_sources.html#arrayrecord-data-source) reads the raw bytes, and we need to integrate a decoder into the data pipeline. SQL Grain's `decode_record` takes care of that, matching the behavior of `to_array_record`.

```python
from sqlgrain import from_array_record, decode_record

ar_source, metadata = from_array_record("output/")
dataset = grain.MapDataset.source(ar_source).map(decode_record).batch(32)
```

`to_array_record` is agnostic to the type of the records, and you can also serialize datasets, e.g., for pre-batching.
