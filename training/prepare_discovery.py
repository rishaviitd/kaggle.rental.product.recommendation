import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from training.config import CONFIG, PipelineConfig
from training.utils import (
    ensure_directory,
    get_logger,
    log_summary,
    load_pickle,
    save_json,
    save_pickle,
    stage_complete,
    stage_start,
)


logger = get_logger(__name__)

SEED_CATEGORY_MAP = {
    "kolyaski": "Коляски",
    "kolyaski-yoyo": "Коляски YoYo",
    "avtokresla": "Автокресла",
    "igrushki": "Игрушки",
    "kacheli-shezlongi": "Электрокачели",
    "videonyani": "Видеоняни",
    "sportkompleksy": "Спортивные комплексы",
}

HIT_COLUMNS = ["client_id", "project_id", "date_time", "slug", "page_type"]


@dataclass(frozen=True)
class DiscoveryArtifacts:
    slug_to_cat_map: dict[str, str]
    search_index: dict[str, list[int]]
    global_top: list[str]


def _project_weight(project_id: Any, new_site_weight: float) -> float:
    try:
        return new_site_weight if int(project_id) == 0 else 1.0
    except (TypeError, ValueError):
        return 1.0


def _update_category_votes(
    hits: pd.DataFrame,
    slug_map: dict[str, int],
    pid_to_cat: dict[int, str],
    slug_votes: defaultdict[str, Counter],
    new_site_weight: float,
) -> None:
    ordered = hits.dropna(subset=["date", "slug"]).sort_values(
        ["client_id", "date"]
    )
    grouped = ordered.groupby(
        ["client_id", "project_id", ordered["date"].dt.date]
    )

    for (_, project_id, _), group in grouped:
        weight = _project_weight(project_id, new_site_weight)
        last_category_slug = None

        for page_type, slug in group[["page_type", "slug"]].itertuples(
            index=False,
            name=None,
        ):
            if page_type == "CATEGORY":
                last_category_slug = slug
                continue

            if page_type != "PRODUCT" or not last_category_slug:
                continue

            product_id = slug_map.get(slug)
            category = pid_to_cat.get(int(product_id)) if product_id else None
            if category:
                slug_votes[last_category_slug][category] += weight


def _update_global_popularity(
    hits: pd.DataFrame,
    slug_map: dict[str, int],
    global_counts: Counter,
    new_site_weight: float,
) -> None:
    product_hits = hits.assign(product_id=hits["slug"].map(slug_map))
    product_hits["product_id"] = pd.to_numeric(
        product_hits["product_id"],
        errors="coerce",
    )
    product_hits = product_hits.dropna(subset=["product_id"])

    for product_id, project_id in product_hits[
        ["product_id", "project_id"]
    ].itertuples(index=False, name=None):
        global_counts[int(product_id)] += _project_weight(
            project_id,
            new_site_weight,
        )


def build_search_index(
    new_products: pd.DataFrame,
    minimum_token_length: int,
) -> dict[str, list[int]]:
    search_index = defaultdict(list)

    for _, row in new_products.iterrows():
        try:
            product_id = int(row["id"])
            text = (
                f"{row['name']} {row['brand']} {row['slug']} "
                f"{row['main_category']}"
            )
            tokens = set(re.split(r"[\s-]+", text.lower().replace("-", " ")))
        except (KeyError, TypeError, ValueError):
            continue

        for token in tokens:
            if len(token) >= minimum_token_length:
                search_index[token].append(product_id)

    return dict(search_index)


def build_discovery_artifacts(
    config: PipelineConfig,
) -> DiscoveryArtifacts:
    intermediate_dir = config.paths.intermediate_dir
    slug_map = load_pickle(intermediate_dir / "slug_map.pkl")
    pid_to_cat = load_pickle(intermediate_dir / "pid_to_cat.pkl")
    new_products = pd.read_parquet(intermediate_dir / "new_products.parquet")

    slug_votes = defaultdict(Counter)
    global_counts = Counter()

    hit_files = ["metrika_hits.parquet", "metrika_hits_test.parquet"]
    for filename in hit_files:
        hits = pd.read_parquet(
            config.paths.data_dir / filename,
            columns=HIT_COLUMNS,
        )
        hits["date"] = pd.to_datetime(hits["date_time"], errors="coerce")
        recent_hits = hits[hits["date"] >= config.discovery.cutoff_date]

        _update_category_votes(
            recent_hits,
            slug_map,
            pid_to_cat,
            slug_votes,
            config.discovery.new_site_weight,
        )
        _update_global_popularity(
            recent_hits,
            slug_map,
            global_counts,
            config.discovery.new_site_weight,
        )
        logger.info(f"  {filename:<28} {len(recent_hits):,} recent hits")

    slug_to_cat_map = dict(SEED_CATEGORY_MAP)
    learned_count = 0
    for slug, votes in slug_votes.items():
        category, count = votes.most_common(1)[0]
        if count > config.discovery.category_min_votes:
            slug_to_cat_map[slug] = category
            learned_count += 1

    search_index = build_search_index(
        new_products,
        config.discovery.search_min_token_length,
    )
    global_top = [
        str(product_id)
        for product_id, _ in global_counts.most_common(
            config.discovery.global_top_k
        )
    ]

    log_summary(
        logger,
        "Discovery Index",
        {
            "Category mappings": f"{len(slug_to_cat_map):,}",
            "Learned mappings": f"{learned_count:,}",
            "Search tokens": f"{len(search_index):,}",
            "Popular products": f"{len(global_counts):,}",
            "Fallback items": f"{len(global_top):,}",
        },
    )

    return DiscoveryArtifacts(
        slug_to_cat_map=slug_to_cat_map,
        search_index=search_index,
        global_top=global_top,
    )


def save_discovery_artifacts(
    artifacts: DiscoveryArtifacts,
    config: PipelineConfig,
) -> None:
    output_dir = ensure_directory(config.paths.intermediate_dir)
    save_pickle(
        output_dir / "slug_to_cat_map.pkl",
        artifacts.slug_to_cat_map,
    )
    save_pickle(output_dir / "search_index.pkl", artifacts.search_index)
    save_json(output_dir / "global_top.json", artifacts.global_top)
    log_summary(
        logger,
        "Output",
        {"Files": "3", "Directory": output_dir},
    )


def prepare_discovery(
    config: PipelineConfig = CONFIG,
) -> DiscoveryArtifacts:
    stage_start(logger, "Prepare Discovery")
    artifacts = build_discovery_artifacts(config)
    save_discovery_artifacts(artifacts, config)
    stage_complete(logger, config.paths.intermediate_dir)
    return artifacts


if __name__ == "__main__":
    prepare_discovery()
