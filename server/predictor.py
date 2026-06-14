from dataclasses import dataclass

import torch

from inference import (
    coocc_recommend,
    config,
    encode_sequence_for_gru,
    load_json,
    load_model,
    load_pickle,
    scored_recommend,
    search_products,
)
from server.database import UserContext


@dataclass(frozen=True)
class Prediction:
    visit_id: str
    product_ids: list[str]
    route: str


class Predictor:
    def __init__(self) -> None:
        print("\n[1/2] Loading inference artifacts")
        self.slug_map = load_pickle("slug_map.pkl")
        self.pid_to_tier = load_pickle("pid_to_tier.pkl")
        self.pid_to_cat_idx = load_pickle("pid_to_cat_idx.pkl")
        self.pid2idx = load_pickle("pid2idx.pkl")
        self.idx2pid = load_pickle("idx2pid.pkl")
        self.p2p = load_pickle("p2p.pkl")
        self.coocc = load_pickle("coocc.pkl")
        self.trigrams = load_pickle("trigrams_dict.pkl")
        self.cat2p = load_pickle("cat2p.pkl")
        self.order_cooccur = load_pickle("order_cooccur.pkl")
        self.search_index = load_pickle("search_index.pkl")
        self.slug_to_cat = load_pickle("slug_to_cat_map.pkl")
        self.global_top = load_json("global_top.json")

        self.device = self._select_device()
        print(f"      Artifacts loaded; device={self.device}")

        print("\n[2/2] Loading Dual GRU model")
        self.model = load_model(
            config["num_items"],
            config["num_categories"],
            self.device,
        )
        print("      Model ready.")

    @staticmethod
    def _select_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def predict(self, context: UserContext) -> Prediction:
        product_sequence: list[int] = []
        gru_actions: list[tuple[str, int, int, int]] = []
        last_slug: str | None = None
        last_category: str | None = None

        for event in context.events:
            if event.slug is not None:
                last_slug = event.slug
            if event.page_type == "CATEGORY" and event.slug is not None:
                last_category = event.slug
            if event.page_type != "PRODUCT" or event.slug is None:
                continue

            product_id = self.slug_map.get(event.slug)
            if product_id is None:
                continue

            product_id = int(product_id)
            product_sequence.append(product_id)
            if not gru_actions or gru_actions[-1][1] != product_id:
                gru_actions.append(
                    (
                        "product",
                        product_id,
                        self.pid_to_tier.get(product_id, 0),
                        self.pid_to_cat_idx.get(product_id, 0),
                    )
                )

        recommendations: list[str] = []
        routes: list[str] = []

        def add_candidates(candidates) -> None:
            for candidate in candidates:
                candidate = str(candidate)
                if candidate not in recommendations:
                    recommendations.append(candidate)
                if len(recommendations) >= 6:
                    break

        if len(product_sequence) in (1, 2):
            add_candidates(
                scored_recommend(
                    product_sequence,
                    self.coocc,
                    self.p2p,
                    self.trigrams,
                    self.cat2p,
                    self.order_cooccur,
                    last_cat=last_category,
                    top_k=6,
                )
            )
            if recommendations:
                routes.append("statistical")

        if len(gru_actions) >= 3 and len(recommendations) < 6:
            encoded = encode_sequence_for_gru(gru_actions, self.pid2idx)
            if len(encoded) >= 2:
                tokens, tiers, categories = map(list, zip(*encoded))
                predicted_indices = self.model.predict_topk(
                    tokens,
                    tiers,
                    categories,
                    k=6,
                    device=self.device,
                    banned=set(tokens),
                )
                add_candidates(
                    self.idx2pid.get(index, index)
                    for index in predicted_indices
                    if index > 0
                )
                if recommendations:
                    routes.append("gru")

        if len(product_sequence) >= 3 and len(recommendations) < 6:
            before = len(recommendations)
            add_candidates(
                coocc_recommend(
                    [str(product_id) for product_id in product_sequence],
                    self.coocc,
                    top_k=6,
                    lookback=5,
                )
            )
            if len(recommendations) > before:
                routes.append("coocc")

        if len(recommendations) < 6 and last_slug is not None:
            before = len(recommendations)
            if last_slug in self.cat2p and self.cat2p[last_slug]:
                add_candidates(
                    product_id
                    for product_id, _ in self.cat2p[last_slug].most_common(10)
                )

            if len(recommendations) < 6:
                query = self.slug_to_cat.get(last_slug, last_slug)
                add_candidates(
                    search_products(query, self.search_index, n=10)
                )

            if len(recommendations) > before:
                routes.append("search")

        if len(recommendations) < 6:
            add_candidates(self.global_top)
            routes.append("global")

        return Prediction(
            visit_id=context.visit_id,
            product_ids=recommendations[:6],
            route="+".join(routes),
        )
