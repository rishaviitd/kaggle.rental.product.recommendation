import ast
from dataclasses import dataclass
from typing import Any

import pandas as pd

from training.config import CONFIG, PipelineConfig
from training.utils import (
    ensure_directory,
    get_logger,
    log_summary,
    save_pickle,
    stage_complete,
    stage_start,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class ProductArtifacts:
    old_to_new: dict[Any, Any]
    slug_map: dict[str, int]
    pid_to_cat: dict[int, str]
    cat2idx: dict[str, int]
    pid_to_cat_idx: dict[int, int]
    pid_to_tier: dict[int, int]
    new_products: pd.DataFrame


def parse_slug_list(value: Any) -> list[str]:
    """Parse the serialized old-slug list used by products_all."""
    if value is None or (not isinstance(value, list) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if pd.notna(item)]
    if value == "":
        return []

    try:
        parsed = ast.literal_eval(value)
    except (TypeError, ValueError, SyntaxError):
        return []

    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if pd.notna(item)]


def build_product_artifacts(config: PipelineConfig) -> ProductArtifacts:
    """Build product mappings from the catalog parquet files."""
    data_dir = config.paths.data_dir

    mapping = pd.read_parquet(data_dir / "old_site_new_site_products.parquet")
    old_to_new = dict(zip(mapping["old_site_id"], mapping["new_site_id"]))

    products_all = pd.read_parquet(data_dir / "products_all.parquet")
    products_all = products_all.dropna(subset=["new_product_id"])

    slug_map: dict[str, int] = {}
    pid_to_cat: dict[int, str] = {}
    pid_to_tier: dict[int, int] = {}
    categories: set[str] = set()

    for _, row in products_all.iterrows():
        try:
            product_id = int(row["new_product_id"])
        except (TypeError, ValueError):
            continue

        new_slug = row.get("new_slug")
        if pd.notna(new_slug) and str(new_slug).strip():
            slug_map[str(new_slug)] = product_id

        for old_slug in parse_slug_list(row.get("old_slugs")):
            if old_slug.strip():
                slug_map[old_slug] = product_id

        category = row.get("main_category")
        if pd.notna(category) and str(category).strip():
            category = str(category)
            pid_to_cat[product_id] = category
            categories.add(category)

        price_tier = row.get("price_tier")
        if pd.notna(price_tier) and str(price_tier).strip():
            pid_to_tier[product_id] = config.products.price_tiers.get(
                str(price_tier).strip(),
                0,
            )

    cat2idx = {
        category: index + 1
        for index, category in enumerate(sorted(categories))
    }
    pid_to_cat_idx = {
        product_id: cat2idx.get(category, 0)
        for product_id, category in pid_to_cat.items()
    }

    old_products = pd.read_parquet(
        data_dir / "old_site_products.parquet",
        columns=["id", "slug"],
    )
    old_products["new_id"] = old_products["id"].map(old_to_new)
    old_products = old_products.dropna(subset=["new_id", "slug"])
    for slug, product_id in zip(
        old_products["slug"],
        old_products["new_id"].astype(int),
    ):
        slug_map.setdefault(slug, product_id)

    new_products = pd.read_parquet(data_dir / "new_site_products.parquet")
    new_products = new_products.dropna(subset=["slug", "id"])
    for slug, product_id in zip(
        new_products["slug"],
        new_products["id"].astype(int),
    ):
        slug_map.setdefault(slug, product_id)

    log_summary(
        logger,
        "Catalog",
        {
            "Products": f"{len(products_all):,}",
            "Old-site mappings": f"{len(old_products):,}",
            "New-site products": f"{len(new_products):,}",
            "Slug mappings": f"{len(slug_map):,}",
            "Categories": f"{len(cat2idx):,}",
            "Products with price tiers": f"{len(pid_to_tier):,}",
        },
    )

    return ProductArtifacts(
        old_to_new=old_to_new,
        slug_map=slug_map,
        pid_to_cat=pid_to_cat,
        cat2idx=cat2idx,
        pid_to_cat_idx=pid_to_cat_idx,
        pid_to_tier=pid_to_tier,
        new_products=new_products,
    )


def save_product_artifacts(
    artifacts: ProductArtifacts,
    config: PipelineConfig,
) -> None:
    """Save Stage 1 outputs for downstream DVC stages."""
    output_dir = ensure_directory(config.paths.intermediate_dir)
    pickle_outputs = {
        "old_to_new.pkl": artifacts.old_to_new,
        "slug_map.pkl": artifacts.slug_map,
        "pid_to_cat.pkl": artifacts.pid_to_cat,
        "cat2idx.pkl": artifacts.cat2idx,
        "pid_to_cat_idx.pkl": artifacts.pid_to_cat_idx,
        "pid_to_tier.pkl": artifacts.pid_to_tier,
    }

    for filename, value in pickle_outputs.items():
        save_pickle(output_dir / filename, value)

    artifacts.new_products.to_parquet(
        output_dir / "new_products.parquet",
        index=False,
    )
    log_summary(
        logger,
        "Output",
        {"Files": "7", "Directory": output_dir},
    )


def prepare_products(config: PipelineConfig = CONFIG) -> ProductArtifacts:
    stage_start(logger, "Prepare Products")
    artifacts = build_product_artifacts(config)
    save_product_artifacts(artifacts, config)
    stage_complete(logger, config.paths.intermediate_dir)
    return artifacts


if __name__ == "__main__":
    prepare_products()
