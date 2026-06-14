import argparse
import asyncio
import os
from pathlib import Path

import asyncpg
import pandas as pd
from dotenv import load_dotenv


USERS_SOURCE_PATH = Path("data/metrika_visits_test.parquet")
EVENTS_SOURCE_PATH = Path("data/metrika_hits_test.parquet")
USER_COLUMNS = ("client_id", "visit_id", "date_time")
EVENT_COLUMNS = ("watch_id", "client_id", "date_time", "page_type", "slug")


def load_users() -> list[tuple[str, str, object]]:
    print("\n[1/3] Reading visit data")
    dataframe = pd.read_parquet(
        USERS_SOURCE_PATH,
        columns=list(USER_COLUMNS),
    )

    if dataframe[list(USER_COLUMNS)].isna().any().any():
        raise ValueError("The users dataset contains null required values.")
    if dataframe["client_id"].duplicated().any():
        raise ValueError("The users dataset contains duplicate client_id values.")

    dataframe["date_time"] = pd.to_datetime(
        dataframe["date_time"],
        format="ISO8601",
        errors="raise",
        utc=True,
    )

    rows = [
        (
            str(row.client_id),
            str(row.visit_id),
            row.date_time.to_pydatetime(),
        )
        for row in dataframe.itertuples(index=False)
    ]
    print(f"      Prepared {len(rows):,} users")
    return rows


def load_events() -> list[tuple[str, str, object, str, str | None]]:
    print("\n[1/3] Reading browsing-event data")
    dataframe = pd.read_parquet(
        EVENTS_SOURCE_PATH,
        columns=list(EVENT_COLUMNS),
    )

    required = ["watch_id", "client_id", "date_time", "page_type"]
    if dataframe[required].isna().any().any():
        raise ValueError("The events dataset contains null required values.")
    if dataframe["watch_id"].duplicated().any():
        raise ValueError("The events dataset contains duplicate watch_id values.")

    dataframe["date_time"] = pd.to_datetime(
        dataframe["date_time"],
        format="ISO8601",
        errors="raise",
        utc=True,
    )

    rows = [
        (
            str(row.watch_id),
            str(row.client_id),
            row.date_time.to_pydatetime(),
            str(row.page_type),
            None if pd.isna(row.slug) else str(row.slug),
        )
        for row in dataframe.itertuples(index=False)
    ]
    print(f"      Prepared {len(rows):,} browsing events")
    return rows


def database_settings() -> dict[str, object]:
    load_dotenv()
    required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing database environment variables: " + ", ".join(missing)
        )

    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ["DB_PORT"]),
        "database": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "ssl": "require",
    }


async def load_users_to_rds() -> None:
    rows = load_users()

    print("\n[2/3] Connecting to PostgreSQL")
    connection = await asyncpg.connect(**database_settings())
    print("      Connected")

    try:
        async with connection.transaction():
            await connection.executemany(
                """
                INSERT INTO users (client_id, visit_id, date_time)
                VALUES ($1, $2, $3)
                ON CONFLICT (client_id) DO UPDATE
                SET visit_id = EXCLUDED.visit_id,
                    date_time = EXCLUDED.date_time
                """,
                rows,
            )

        print("\n[3/3] Verifying upload")
        total = await connection.fetchval("SELECT COUNT(*) FROM users")
        print(f"      Users in RDS: {total:,}")
        print("\nUsers upload complete.")
    finally:
        await connection.close()


async def load_events_to_rds() -> None:
    rows = load_events()

    print("\n[2/3] Connecting to PostgreSQL")
    connection = await asyncpg.connect(**database_settings())
    print("      Connected")

    try:
        async with connection.transaction():
            await connection.executemany(
                """
                INSERT INTO user_browsing_events (
                    event_id,
                    client_id,
                    date_time,
                    page_type,
                    slug
                )
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (event_id) DO UPDATE
                SET client_id = EXCLUDED.client_id,
                    date_time = EXCLUDED.date_time,
                    page_type = EXCLUDED.page_type,
                    slug = EXCLUDED.slug
                """,
                rows,
            )

        print("\n[3/3] Verifying upload")
        total = await connection.fetchval(
            "SELECT COUNT(*) FROM user_browsing_events"
        )
        print(f"      Browsing events in RDS: {total:,}")
        print("\nBrowsing-events upload complete.")
    finally:
        await connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load inference datasets into RDS."
    )
    parser.add_argument(
        "--dataset",
        choices=("users", "events"),
        required=True,
        help="Dataset to upload.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "users":
        asyncio.run(load_users_to_rds())
    else:
        asyncio.run(load_events_to_rds())
