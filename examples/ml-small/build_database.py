import argparse
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pandas as pd


class _Args:
    schema: Path
    movielens: Path
    database: Path


def __main__() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema",
        type=Path,
        help="Path to database schema definition.",
        default="schema.sql",
    )
    parser.add_argument(
        "movielens", type=Path, help="Path to movielens data directory."
    )
    parser.add_argument("database", type=Path, help="Path to target database.")
    args = cast(_Args, parser.parse_args())

    assert args.schema.is_file(), (
        f"'{args.schema}' is not a file. Use '--schema' to specify a path to a schema definition."
    )
    assert args.movielens.is_dir(), f"'{args.movielens}' is not a directory."
    assert not args.database.exists(), f"'{args.database}' already exists."

    # We use with `closing` to close the connection gracefully and with `conn` to wrap
    # the code in a transaction.
    with closing(sqlite3.connect(args.database)) as conn, conn:
        conn.executescript(args.schema.read_text())

        # Insert all movies.
        movies = pd.read_csv(
            args.movielens / "movies.dat",
            header=None,
            names=["id", "title", "genres"],
            sep="::",
            encoding="latin-1",
            engine="python",
        )
        cursor = conn.executemany(
            "INSERT INTO movies (id, title) VALUES (:id, :title)",
            movies.to_dict(orient="records"),
        )
        print(f"🎬 Inserted {cursor.rowcount:,} movies.")

        # Insert all users.
        users = pd.read_csv(
            args.movielens / "users.dat",
            header=None,
            names=["id", "gender", "age", "occupation", "zip"],
            sep="::",
            encoding="latin-1",
            engine="python",
        )
        cursor = conn.executemany(
            "INSERT INTO users (id, gender, age) VALUES (:id, :gender, :age)",
            users.to_dict(orient="records"),
        )
        print(f"👥 Inserted {cursor.rowcount:,} users.")

        # Insert all ratings.
        ratings = pd.read_csv(
            args.movielens / "ratings.dat",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            sep="::",
            encoding="latin-1",
            engine="python",
        )
        cursor = conn.executemany(
            """
            INSERT INTO ratings (user_id, movie_id, rating, timestamp)
            VALUES (:user_id, :movie_id, :rating, :timestamp)
            """,
            ratings.to_dict(orient="records"),
        )
        print(f"⭐️ Inserted {cursor.rowcount:,} ratings.")


if __name__ == "__main__":
    __main__()
