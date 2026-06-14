import argparse
import asyncio
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from server.database import Database
from server.predictor import Predictor


DEFAULT_OUTPUT = Path("output/predictions.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Kaggle predictions from inference data in RDS."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


async def generate_predictions(output_path: Path) -> Path:
    print("\n=== RDS BATCH INFERENCE ===")
    database = Database()
    await database.connect()

    try:
        contexts = await database.fetch_all_user_contexts()
    finally:
        await database.disconnect()

    predictor = Predictor()
    print(f"\nPredicting {len(contexts):,} visits...")

    rows = []
    route_counts: Counter[str] = Counter()
    for context in tqdm(contexts):
        prediction = predictor.predict(context)
        rows.append(
            {
                "visit_id": prediction.visit_id,
                "product_ids": " ".join(prediction.product_ids),
            }
        )
        route_counts[prediction.route] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\nSaved {len(rows):,} predictions to {output_path}")
    print("Routes:")
    for route, count in route_counts.most_common():
        print(f"  {route:<24} {count:>5,}")
    return output_path


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(generate_predictions(arguments.output))
