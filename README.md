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
