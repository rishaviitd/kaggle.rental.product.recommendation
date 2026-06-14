import os
from dataclasses import dataclass
from datetime import datetime

import asyncpg
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class BrowsingEvent:
    event_id: str
    date_time: datetime
    page_type: str
    slug: str | None


@dataclass(frozen=True)
class UserContext:
    client_id: str
    visit_id: str
    visit_start: datetime
    events: list[BrowsingEvent]


def database_settings() -> dict[str, object]:
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
        "min_size": 1,
        "max_size": 5,
    }


class Database:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return

        print("Connecting to PostgreSQL...")
        self._pool = await asyncpg.create_pool(**database_settings())
        print("PostgreSQL connection pool ready.")

    async def disconnect(self) -> None:
        if self._pool is None:
            return

        await self._pool.close()
        self._pool = None
        print("PostgreSQL connection pool closed.")

    async def fetch_user_context(
        self,
        client_id: str,
    ) -> UserContext | None:
        if self._pool is None:
            raise RuntimeError("Database connection pool is not initialized.")

        rows = await self._pool.fetch(
            """
            SELECT
                u.client_id,
                u.visit_id,
                u.date_time AS visit_start,
                e.event_id,
                e.date_time AS event_time,
                e.page_type,
                e.slug
            FROM users AS u
            LEFT JOIN user_browsing_events AS e
                ON e.client_id = u.client_id
                AND e.date_time >= u.date_time
                AND e.date_time <= u.date_time + INTERVAL '2 hours'
            WHERE u.client_id = $1
            ORDER BY e.date_time ASC, e.event_id ASC
            """,
            client_id,
        )

        if not rows:
            return None

        return self._build_user_context(rows)

    async def fetch_all_user_contexts(self) -> list[UserContext]:
        if self._pool is None:
            raise RuntimeError("Database connection pool is not initialized.")

        print("Fetching all user contexts from PostgreSQL...")
        rows = await self._pool.fetch(
            """
            SELECT
                u.client_id,
                u.visit_id,
                u.date_time AS visit_start,
                e.event_id,
                e.date_time AS event_time,
                e.page_type,
                e.slug
            FROM users AS u
            LEFT JOIN user_browsing_events AS e
                ON e.client_id = u.client_id
                AND e.date_time >= u.date_time
                AND e.date_time <= u.date_time + INTERVAL '2 hours'
            ORDER BY
                u.date_time ASC,
                u.client_id ASC,
                e.date_time ASC,
                e.event_id ASC
            """
        )

        grouped_rows: dict[str, list[asyncpg.Record]] = {}
        for row in rows:
            grouped_rows.setdefault(str(row["client_id"]), []).append(row)

        contexts = [
            self._build_user_context(user_rows)
            for user_rows in grouped_rows.values()
        ]
        print(f"Fetched {len(contexts):,} user contexts.")
        return contexts

    @staticmethod
    def _build_user_context(rows: list[asyncpg.Record]) -> UserContext:
        events = [
            BrowsingEvent(
                event_id=str(row["event_id"]),
                date_time=row["event_time"],
                page_type=row["page_type"],
                slug=row["slug"],
            )
            for row in rows
            if row["event_id"] is not None
        ]

        first = rows[0]
        return UserContext(
            client_id=str(first["client_id"]),
            visit_id=str(first["visit_id"]),
            visit_start=first["visit_start"],
            events=events,
        )
