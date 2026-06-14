import argparse
import asyncio

from server.database import Database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one user's inference context from RDS."
    )
    parser.add_argument("client_id", help="Client identifier to query.")
    return parser.parse_args()


async def show_user_context(client_id: str) -> None:
    database = Database()
    await database.connect()

    try:
        print(f"\nFetching context for client {client_id}...")
        context = await database.fetch_user_context(client_id)

        if context is None:
            print("Client not found.")
            return

        print("\nUser Context")
        print(f"  Client ID    {context.client_id}")
        print(f"  Visit ID     {context.visit_id}")
        print(f"  Visit start  {context.visit_start}")
        print(f"  Events       {len(context.events):,}")

        print("\nOrdered Events")
        if not context.events:
            print("  No events found in the two-hour visit window.")
            return

        for index, event in enumerate(context.events, start=1):
            print(
                f"  {index:>2}. {event.date_time}  "
                f"{event.page_type:<10}  {event.slug or '-'}"
            )
    finally:
        await database.disconnect()


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(show_user_context(arguments.client_id))
