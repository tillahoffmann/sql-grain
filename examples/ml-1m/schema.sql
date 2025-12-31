CREATE TABLE movies (
    "id" INTEGER PRIMARY KEY,
    "title" TEXT NOT NULL
);

CREATE TABLE users (
    "id" INTEGER PRIMARY KEY,
    "gender" TEXT NOT NULL,
    "age" INTEGER NOT NULL
);

CREATE TABLE ratings (
    "movie_id" INTEGER NOT NULL,
    "user_id" INTEGER NOT NULL,
    "rating" INTEGER NOT NULL,
    "timestamp" INTEGER NOT NULL,
    PRIMARY KEY ("user_id", "movie_id"),
    FOREIGN KEY ("movie_id") REFERENCES "movies"("id"),
    FOREIGN KEY ("user_id") REFERENCES "users"("id")
);

-- Covering index for efficiently looking up by user id and sorting by timestamp. Including
-- movie_id makes this a covering index, avoiding table lookups when only movie_id is needed.
CREATE INDEX idx_ratings_user_timestamp ON ratings ("user_id", "timestamp" DESC, "movie_id");
