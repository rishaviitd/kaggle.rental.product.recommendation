import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm


ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data")
DEFAULT_OUTPUT = Path("output/predictions.csv")
REFERENCE_OUTPUT = Path("output/submission.csv")


def load_pickle(name):
    with (ARTIFACTS_DIR / name).open("rb") as f:
        return pickle.load(f)


def load_json(name):
    with (ARTIFACTS_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


config = load_json("config.json")

PAD_IDX = config["pad_idx"]
SESSION_TOLERANCE = pd.Timedelta(config["session_tolerance"])
USE_DWELL_TIME = config["use_dwell_time"]
USE_SESSION_ELAPSED = config["use_session_elapsed"]
USE_TIME_OF_DAY = config["use_time_of_day"]
USE_DAY_OF_WEEK = config["use_day_of_week"]
USE_TIME_DECAY = config["use_time_decay"]
USE_MONTH = config["use_month"]
USE_INTER_SESSION_GAP = config["use_inter_session_gap"]
USE_PRODUCT_AGE = config["use_product_age"]
USE_PRODUCT_RECENCY = config["use_product_recency"]
USE_PRODUCT_VELOCITY = config["use_product_velocity"]
MAX_DWELL_SECONDS = config["max_dwell_seconds"]
MAX_SESSION_SECONDS = config["max_session_seconds"]
MAX_GAP_DAYS = config["max_gap_days"]
MAX_PRODUCT_AGE_DAYS = config["max_product_age_days"]
MAX_PRODUCT_RECENCY_DAYS = config["max_product_recency_days"]
MAX_PRODUCT_VELOCITY = config["max_product_velocity"]


def log_norm(value, max_value):
    if max_value <= 0:
        return 0.0
    value = max(0.0, min(float(value), float(max_value)))
    return np.log1p(value) / np.log1p(max_value)


def build_sessions_merge_asof(hits_path, visits_path, slug_map, tolerance=SESSION_TOLERANCE):
    print(f"   Loading hits from {hits_path}...")
    hits = pd.read_parquet(hits_path)
    hits["date_time"] = pd.to_datetime(hits["date_time"], format="ISO8601", errors="coerce")
    hits = hits.dropna(subset=["date_time", "client_id"])
    print(f"   Loaded {len(hits):,} hits")

    print(f"   Loading visits from {visits_path}...")
    visits = pd.read_parquet(visits_path)
    visits["date_time"] = pd.to_datetime(visits["date_time"], format="ISO8601", errors="coerce")
    visits = visits.dropna(subset=["date_time", "client_id", "visit_id"])
    print(f"   Loaded {len(visits):,} visits")

    visits = visits.rename(columns={"date_time": "visit_start"})
    hits_sorted = hits.sort_values(["client_id", "date_time"]).reset_index(drop=True)
    visits_sorted = visits.sort_values(["client_id", "visit_start"]).reset_index(drop=True)

    print(f"   Performing merge_asof with {tolerance} tolerance...")
    merged = pd.merge_asof(
        hits_sorted.sort_values("date_time"),
        visits_sorted.sort_values("visit_start")[["client_id", "visit_start", "visit_id", "project_id"]],
        by="client_id",
        left_on="date_time",
        right_on="visit_start",
        direction="backward",
        tolerance=tolerance,
    )

    matched = merged["visit_id"].notna().sum()
    total = len(merged)
    print(f"   Matched {matched:,}/{total:,} hits ({matched / total * 100:.2f}%)")
    return merged


def extract_product_sequences_from_merged(merged_df, slug_map):
    products = merged_df[merged_df["page_type"] == "PRODUCT"].copy()
    products["product_id"] = products["slug"].map(slug_map)
    products = products.dropna(subset=["visit_id", "product_id"])
    products["product_id"] = products["product_id"].astype(int).astype(str)

    sequences = {}
    for visit_id, group in products.groupby("visit_id"):
        seq = group.sort_values("date_time")["product_id"].tolist()
        if seq:
            sequences[str(visit_id)] = seq
    return sequences


def get_last_slug_from_merged(merged_df):
    relevant = merged_df.dropna(subset=["visit_id", "slug"])
    last_hits = relevant.sort_values("date_time").groupby("visit_id").last()
    return last_hits["slug"].to_dict()


def get_last_cat_from_merged(merged_df):
    cats = merged_df[merged_df["page_type"] == "CATEGORY"].copy()
    cats = cats.dropna(subset=["visit_id", "slug"])
    if len(cats) == 0:
        return {}
    last_cats = cats.sort_values("date_time").groupby("visit_id").last()
    return last_cats["slug"].to_dict()


def coocc_recommend(seq, coocc, top_k=6, lookback=5):
    if len(seq) < 1:
        return []
    recent = seq[-lookback:]
    scores = defaultdict(float)
    for i, item in enumerate(reversed(recent)):
        pos_weight = 1.0 / (i + 1)
        for neighbor, w in coocc.get(item, {}).items():
            scores[neighbor] += pos_weight * w
    if not scores:
        return []
    topk = [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:50]]
    seen = set(seq)
    recs = []
    for pid in topk:
        if pid in seen:
            continue
        recs.append(pid)
        if len(recs) >= top_k:
            break
    return recs


def scored_recommend(product_seq, coocc, p2p, trigrams_dict, cat2p_dict, order_cooccur, last_cat=None, top_k=6):
    seen = set(str(x) for x in product_seq)
    cand = defaultdict(float)
    str_seq = [str(x) for x in product_seq]

    for i, item in enumerate(reversed(str_seq[-5:])):
        pos_weight = 1.0 / (i + 1)
        for neighbor, w in coocc.get(item, {}).items():
            cand[neighbor] += pos_weight * w

    for k, v in p2p.get(str_seq[-1], Counter()).most_common(20):
        cand[k] += v * 2.0

    if len(str_seq) >= 2:
        key = (str_seq[-2], str_seq[-1])
        for k, v in trigrams_dict.get(key, Counter()).most_common(20):
            cand[k] += v * 3.0

    if last_cat and last_cat in cat2p_dict:
        for k, v in cat2p_dict[last_cat].most_common(20):
            cand[k] += v * 1.5

    for p in str_seq[-2:]:
        for k, v in order_cooccur.get(p, Counter()).most_common(10):
            cand[k] += v * 2.0

    ranked = sorted(cand.items(), key=lambda x: -x[1])
    recs = []
    for pid, _ in ranked:
        if pid not in seen:
            recs.append(pid)
            if len(recs) >= top_k:
                break
    return recs


def deduplicate_consecutive_with_temporal(actions):
    if not actions:
        return []
    deduped = [actions[0]]
    for i in range(1, len(actions)):
        (
            curr_type,
            curr_pid,
            curr_dwell,
            curr_elapsed,
            curr_hour,
            curr_dow,
            curr_month,
            curr_gap,
            curr_age,
            curr_recency,
            curr_velocity,
            curr_tier,
            curr_pop,
            curr_nb,
            curr_cat,
        ) = actions[i]
        (
            prev_type,
            prev_pid,
            prev_dwell,
            prev_elapsed,
            prev_hour,
            prev_dow,
            prev_month,
            prev_gap,
            prev_age,
            prev_recency,
            prev_velocity,
            prev_tier,
            prev_pop,
            prev_nb,
            prev_cat,
        ) = deduped[-1]
        if curr_type == prev_type and curr_pid == prev_pid:
            deduped[-1] = (
                prev_type,
                prev_pid,
                prev_dwell + curr_dwell,
                prev_elapsed,
                prev_hour,
                prev_dow,
                prev_month,
                prev_gap,
                prev_age,
                prev_recency,
                prev_velocity,
                prev_tier,
                prev_pop,
                max(prev_nb, curr_nb),
                prev_cat,
            )
        else:
            deduped.append(actions[i])
    return deduped


def extract_actions_with_temporal(
    merged_df,
    slug_map,
    visit_gap_map,
    product_first_seen,
    product_recency,
    velocity_by_pid,
    velocity_mean,
    pid_to_tier,
    pid_to_pop,
    pid_to_cat_idx,
):
    relevant = merged_df[merged_df["page_type"] == "PRODUCT"].copy()
    relevant = relevant.dropna(subset=["visit_id", "slug"])
    relevant["product_id"] = relevant["slug"].map(slug_map)
    relevant = relevant.dropna(subset=["product_id"])

    results = []
    for visit_id, group in relevant.groupby("visit_id"):
        sorted_group = group.sort_values("date_time").reset_index(drop=True)
        actions = []
        timestamps = sorted_group["date_time"].tolist()
        pids = sorted_group["product_id"].tolist()
        if "not_bounce" in sorted_group.columns:
            not_bounce_vals = sorted_group["not_bounce"].fillna(0).astype(int).tolist()
        else:
            not_bounce_vals = [0] * len(pids)
        session_start = timestamps[0]
        gap_days = visit_gap_map.get(str(visit_id), 0.0) if USE_INTER_SESSION_GAP else 0.0

        for i in range(len(pids)):
            if i < len(timestamps) - 1:
                dwell_seconds = (timestamps[i + 1] - timestamps[i]).total_seconds()
                dwell_seconds = min(max(dwell_seconds, 0), MAX_DWELL_SECONDS)
            else:
                dwell_seconds = 0

            elapsed_seconds = (timestamps[i] - session_start).total_seconds()
            elapsed_seconds = min(max(elapsed_seconds, 0), MAX_SESSION_SECONDS)
            pid = int(pids[i])
            action_date = timestamps[i].date()

            first_seen_date = product_first_seen.get(pid)
            if first_seen_date is None:
                age_days = 0.0
            else:
                age_days = max((action_date - first_seen_date).days, 0.0)

            recency_days = product_recency.get(pid, 0.0)
            velocity = velocity_by_pid.get(pid, {}).get(action_date)
            if velocity is None:
                velocity = velocity_mean

            actions.append(
                (
                    "product",
                    pid,
                    dwell_seconds,
                    elapsed_seconds,
                    timestamps[i].hour,
                    timestamps[i].dayofweek,
                    timestamps[i].month,
                    gap_days,
                    age_days,
                    recency_days,
                    velocity,
                    pid_to_tier.get(pid, 0),
                    pid_to_pop.get(pid, 0.0),
                    int(not_bounce_vals[i]) if i < len(not_bounce_vals) else 0,
                    pid_to_cat_idx.get(pid, 0),
                )
            )

        actions = deduplicate_consecutive_with_temporal(actions)
        if actions:
            results.append(
                {
                    "session_id": str(visit_id),
                    "user_actions": actions,
                    "timestamp": sorted_group["date_time"].iloc[0],
                }
            )

    return pd.DataFrame(results)


def encode_sequence_with_temporal(actions, pid2idx, p2p=None, p2p_totals=None):
    result = []
    T = len(actions)
    for i, (
        typ,
        pid,
        dwell,
        elapsed,
        hour,
        dow,
        month,
        gap_days,
        age_days,
        recency_days,
        velocity,
        tier_idx,
        pop_score,
        not_bounce,
        cat_idx,
    ) in enumerate(actions):
        token_id = pid2idx.get(pid, 0)
        if token_id > 0:
            norm_dwell = np.log1p(dwell) / np.log1p(MAX_DWELL_SECONDS)
            norm_elapsed = np.log1p(elapsed) / np.log1p(MAX_SESSION_SECONDS)
            hour_sin = np.sin(2 * np.pi * hour / 24.0)
            hour_cos = np.cos(2 * np.pi * hour / 24.0)
            dow_idx = dow
            month_sin = np.sin(2 * np.pi * month / 12.0) if USE_MONTH else 0.0
            month_cos = np.cos(2 * np.pi * month / 12.0) if USE_MONTH else 0.0
            gap_norm = log_norm(gap_days, MAX_GAP_DAYS) if USE_INTER_SESSION_GAP else 0.0
            age_norm = log_norm(age_days, MAX_PRODUCT_AGE_DAYS) if USE_PRODUCT_AGE else 0.0
            recency_norm = log_norm(recency_days, MAX_PRODUCT_RECENCY_DAYS) if USE_PRODUCT_RECENCY else 0.0
            vel_norm = log_norm(velocity, MAX_PRODUCT_VELOCITY) if USE_PRODUCT_VELOCITY else 0.0
            decay = np.exp(-0.1 * (T - 1 - i))

            p2p_score = 0.0
            if p2p is not None and p2p_totals is not None and i > 0:
                prev_pid = actions[i - 1][1]
                prev_key = str(prev_pid)
                curr_key = str(pid)
                count = p2p.get(prev_key, {}).get(curr_key, 0)
                total = p2p_totals.get(prev_key, 0)
                if total > 0:
                    p2p_score = float(count) / float(total)

            result.append(
                (
                    token_id,
                    norm_dwell,
                    norm_elapsed,
                    hour_sin,
                    hour_cos,
                    dow_idx,
                    month_sin,
                    month_cos,
                    gap_norm,
                    age_norm,
                    recency_norm,
                    vel_norm,
                    decay,
                    int(tier_idx),
                    float(pop_score),
                    float(p2p_score),
                    float(not_bounce),
                    int(cat_idx),
                )
            )
    return result


class GRURecDual(nn.Module):
    def __init__(
        self,
        num_items,
        num_categories,
        item_emb_dim=128,
        tier_emb_dim=4,
        cat_emb_dim=8,
        hidden_dim=128,
        cat_hidden_dim=96,
        num_layers=1,
        dropout=0.2,
        use_dwell=True,
        use_elapsed=True,
        use_tod=True,
        use_dow=True,
        use_decay=True,
        use_month=True,
        use_gap=True,
        use_age=True,
        use_recency=True,
        use_velocity=True,
        use_pop=True,
        use_p2p=True,
        use_not_bounce=True,
    ):
        super().__init__()
        self.use_dwell = use_dwell
        self.use_elapsed = use_elapsed
        self.use_tod = use_tod
        self.use_dow = use_dow
        self.use_decay = use_decay
        self.use_month = use_month
        self.use_gap = use_gap
        self.use_age = use_age
        self.use_recency = use_recency
        self.use_velocity = use_velocity
        self.use_pop = use_pop
        self.use_p2p = use_p2p
        self.use_not_bounce = use_not_bounce

        self.item_emb = nn.Embedding(num_items + 1, item_emb_dim, padding_idx=PAD_IDX)
        nn.init.xavier_uniform_(self.item_emb.weight.data)
        self.tier_emb = nn.Embedding(6, tier_emb_dim, padding_idx=PAD_IDX)
        self.cat_emb = nn.Embedding(num_categories + 1, cat_emb_dim, padding_idx=PAD_IDX)
        if use_dow:
            self.dow_emb = nn.Embedding(7, 4)

        in_dim = item_emb_dim + tier_emb_dim
        if use_dwell:
            in_dim += 1
        if use_elapsed:
            in_dim += 1
        if use_tod:
            in_dim += 2
        if use_dow:
            in_dim += 4
        if use_month:
            in_dim += 2
        if use_gap:
            in_dim += 1
        if use_age:
            in_dim += 1
        if use_recency:
            in_dim += 1
        if use_velocity:
            in_dim += 1
        if use_decay:
            in_dim += 1
        if use_pop:
            in_dim += 1
        if use_p2p:
            in_dim += 1
        if use_not_bounce:
            in_dim += 1

        self.item_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.item_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.cat_proj = nn.Sequential(nn.Linear(cat_emb_dim, cat_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.cat_gru = nn.GRU(
            input_size=cat_hidden_dim,
            hidden_size=cat_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim + cat_hidden_dim, num_items + 1)

    def forward(
        self,
        x,
        tier=None,
        pop=None,
        p2p_score=None,
        not_bounce=None,
        cat=None,
        dwell=None,
        elapsed=None,
        hour=None,
        dow=None,
        month=None,
        gap=None,
        age=None,
        recency=None,
        velocity=None,
        decay=None,
    ):
        e = self.item_emb(x)
        t = self.tier_emb(tier) if tier is not None else torch.zeros_like(e[..., :0])

        feats_list = [e, t]
        if self.use_dwell and dwell is not None:
            feats_list.append(dwell.unsqueeze(-1))
        if self.use_elapsed and elapsed is not None:
            feats_list.append(elapsed.unsqueeze(-1))
        if self.use_tod and hour is not None:
            feats_list.append(hour)
        if self.use_dow and dow is not None:
            feats_list.append(self.dow_emb(dow))
        if self.use_month and month is not None:
            feats_list.append(month)
        if self.use_gap and gap is not None:
            feats_list.append(gap.unsqueeze(-1))
        if self.use_age and age is not None:
            feats_list.append(age.unsqueeze(-1))
        if self.use_recency and recency is not None:
            feats_list.append(recency.unsqueeze(-1))
        if self.use_velocity and velocity is not None:
            feats_list.append(velocity.unsqueeze(-1))
        if self.use_decay and decay is not None:
            feats_list.append(decay.unsqueeze(-1))
        if self.use_pop and pop is not None:
            feats_list.append(pop.unsqueeze(-1))
        if self.use_p2p and p2p_score is not None:
            feats_list.append(p2p_score.unsqueeze(-1))
        if self.use_not_bounce and not_bounce is not None:
            feats_list.append(not_bounce.unsqueeze(-1))

        item_h = self.item_proj(torch.cat(feats_list, dim=-1))
        item_h, _ = self.item_gru(item_h)

        if cat is None:
            cat = torch.zeros_like(x)
        cat_h = self.cat_proj(self.cat_emb(cat))
        cat_h, _ = self.cat_gru(cat_h)

        return self.out(torch.cat([item_h, cat_h], dim=-1))

    @torch.no_grad()
    def predict_topk(
        self,
        session_tokens,
        session_tier,
        session_pop,
        session_p2p,
        session_not_bounce,
        session_cat,
        session_dwell,
        session_elapsed,
        session_hour,
        session_dow,
        session_month,
        session_gap,
        session_age,
        session_recency,
        session_velocity,
        session_decay,
        k=6,
        device="cpu",
        banned=None,
    ):
        if not session_tokens:
            return []

        x = torch.tensor(session_tokens, dtype=torch.long, device=device).unsqueeze(0)
        tier_t = torch.tensor(session_tier, dtype=torch.long, device=device).unsqueeze(0)
        pop_t = torch.tensor(session_pop, dtype=torch.float, device=device).unsqueeze(0)
        p2p_t = torch.tensor(session_p2p, dtype=torch.float, device=device).unsqueeze(0)
        nb_t = torch.tensor(session_not_bounce, dtype=torch.float, device=device).unsqueeze(0)
        cat_t = torch.tensor(session_cat, dtype=torch.long, device=device).unsqueeze(0)
        dwell_t = torch.tensor(session_dwell, dtype=torch.float, device=device).unsqueeze(0)
        elapsed_t = torch.tensor(session_elapsed, dtype=torch.float, device=device).unsqueeze(0)
        hour_t = torch.tensor(session_hour, dtype=torch.float, device=device).unsqueeze(0)
        dow_t = torch.tensor(session_dow, dtype=torch.long, device=device).unsqueeze(0)
        month_t = torch.tensor(session_month, dtype=torch.float, device=device).unsqueeze(0)
        gap_t = torch.tensor(session_gap, dtype=torch.float, device=device).unsqueeze(0)
        age_t = torch.tensor(session_age, dtype=torch.float, device=device).unsqueeze(0)
        recency_t = torch.tensor(session_recency, dtype=torch.float, device=device).unsqueeze(0)
        velocity_t = torch.tensor(session_velocity, dtype=torch.float, device=device).unsqueeze(0)
        decay_t = torch.tensor(session_decay, dtype=torch.float, device=device).unsqueeze(0)

        logits = self.forward(
            x,
            tier_t,
            pop_t,
            p2p_t,
            nb_t,
            cat_t,
            dwell_t,
            elapsed_t,
            hour_t,
            dow_t,
            month_t,
            gap_t,
            age_t,
            recency_t,
            velocity_t,
            decay_t,
        )[0, -1]

        if banned:
            for b in banned:
                if 0 < b < len(logits):
                    logits[b] = -float("inf")

        logits[0] = -float("inf")
        _, top_indices = torch.topk(logits, min(k + len(session_tokens), len(logits)))

        result = []
        for idx in top_indices.tolist():
            if idx > 0 and idx not in session_tokens:
                result.append(idx)
                if len(result) >= k:
                    break
        return result


def search_products(query_slug, search_index, n=10):
    if pd.isna(query_slug):
        return []
    keywords = [k for k in str(query_slug).lower().replace("-", " ").split() if len(k) > 2]
    if not keywords:
        return []
    candidates = Counter()
    for k in keywords:
        if k in search_index:
            for pid in search_index[k]:
                candidates[pid] += 1
    return [pid for pid, _ in candidates.most_common(n)]


def load_model(num_items, num_categories, device):
    model = GRURecDual(
        num_items=num_items,
        num_categories=num_categories,
        use_dwell=USE_DWELL_TIME,
        use_elapsed=USE_SESSION_ELAPSED,
        use_tod=USE_TIME_OF_DAY,
        use_dow=USE_DAY_OF_WEEK,
        use_decay=USE_TIME_DECAY,
        use_month=USE_MONTH,
        use_gap=USE_INTER_SESSION_GAP,
        use_age=USE_PRODUCT_AGE,
        use_recency=USE_PRODUCT_RECENCY,
        use_velocity=USE_PRODUCT_VELOCITY,
        use_pop=True,
        use_p2p=True,
        use_not_bounce=True,
    )
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model


def generate_submission(output_path):
    print("Loading artifacts...")
    old_to_new = load_pickle("old_to_new.pkl")
    slug_map = load_pickle("slug_map.pkl")
    pid_to_tier = load_pickle("pid_to_tier.pkl")
    pid_to_pop = load_pickle("pid_to_pop.pkl")
    pid_to_cat_idx = load_pickle("pid_to_cat_idx.pkl")
    pid2idx = load_pickle("pid2idx.pkl")
    idx2pid = load_pickle("idx2pid.pkl")
    product_first_seen = load_pickle("product_first_seen.pkl")
    product_recency = load_pickle("product_recency.pkl")
    velocity_by_pid = load_pickle("velocity_by_pid.pkl")
    p2p = load_pickle("p2p.pkl")
    p2p_totals = load_pickle("p2p_totals.pkl")
    coocc = load_pickle("coocc.pkl")
    trigrams_dict = load_pickle("trigrams_dict.pkl")
    cat2p = load_pickle("cat2p.pkl")
    order_cooccur = load_pickle("order_cooccur.pkl")
    search_index = load_pickle("search_index.pkl")
    slug_to_cat_map = load_pickle("slug_to_cat_map.pkl")
    visit_gap_map = load_pickle("visit_gap_map.pkl")
    global_top = load_json("global_top.json")

    _ = old_to_new
    velocity_mean = config["velocity_mean"]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    gru_model = load_model(config["num_items"], config["num_categories"], device)

    print("Building test sessions from raw test data...")
    test_merged = build_sessions_merge_asof(
        DATA_DIR / "metrika_hits_test.parquet",
        DATA_DIR / "metrika_visits_test.parquet",
        slug_map,
    )
    test_sequences_by_visit = extract_product_sequences_from_merged(test_merged, slug_map)

    print("Extracting and encoding temporal test actions...")
    test_actions = extract_actions_with_temporal(
        test_merged,
        slug_map,
        visit_gap_map,
        product_first_seen,
        product_recency,
        velocity_by_pid,
        velocity_mean,
        pid_to_tier,
        pid_to_pop,
        pid_to_cat_idx,
    )
    test_actions["product_count"] = test_actions["user_actions"].apply(len)

    test_encoded = {}
    for _, row in test_actions.iterrows():
        vid = str(row["session_id"])
        if row["product_count"] >= 3:
            encoded = encode_sequence_with_temporal(
                row["user_actions"],
                pid2idx,
                p2p=p2p,
                p2p_totals=p2p_totals,
            )
            if len(encoded) >= 2:
                test_encoded[vid] = encoded

    print("Building test context...")
    last_slug_by_visit = get_last_slug_from_merged(test_merged)
    last_cat_by_visit = get_last_cat_from_merged(test_merged)
    test_context = {}
    for vid, seq in test_sequences_by_visit.items():
        test_context[vid] = {
            "last_slug": last_slug_by_visit.get(vid),
            "last_cat": last_cat_by_visit.get(vid),
            "product_seq": [int(pid) for pid in seq],
        }
    for vid, slug in last_slug_by_visit.items():
        vid = str(vid)
        if vid not in test_context:
            test_context[vid] = {
                "last_slug": slug,
                "last_cat": last_cat_by_visit.get(vid),
                "product_seq": [],
            }

    targets = pd.read_parquet(DATA_DIR / "metrika_visits_test.parquet", columns=["visit_id"])
    print(f"Predicting {len(targets):,} visits...")

    preds = []
    stats = {"coocc": 0, "gru": 0, "search": 0, "global": 0}
    for vid in tqdm(targets["visit_id"]):
        recs = []

        def add_candidates(candidates):
            for cand in candidates:
                if cand not in recs:
                    recs.append(cand)
                if len(recs) >= 6:
                    break

        if vid in test_context:
            action = test_context[vid]
            slug = action["last_slug"]
            product_seq = action["product_seq"]

            if len(product_seq) in [1, 2] and len(recs) < 6:
                last_cat = action.get("last_cat")
                add_candidates(
                    scored_recommend(
                        product_seq,
                        coocc,
                        p2p,
                        trigrams_dict,
                        cat2p,
                        order_cooccur,
                        last_cat=last_cat,
                        top_k=6,
                    )
                )
                if recs:
                    stats["coocc"] += 1

            if len(product_seq) >= 3 and len(recs) < 6:
                if vid in test_encoded:
                    encoded_seq = test_encoded[vid]
                    unpacked = list(zip(*encoded_seq))
                    session_tokens = list(unpacked[0])
                    session_dwell = list(unpacked[1])
                    session_elapsed = list(unpacked[2])
                    session_hour = list(zip(unpacked[3], unpacked[4]))
                    session_dow = list(unpacked[5])
                    session_month = list(zip(unpacked[6], unpacked[7]))
                    session_gap = list(unpacked[8])
                    session_age = list(unpacked[9])
                    session_recency = list(unpacked[10])
                    session_velocity = list(unpacked[11])
                    session_decay = list(unpacked[12])
                    session_tier = list(unpacked[13])
                    session_pop = list(unpacked[14])
                    session_p2p = list(unpacked[15])
                    session_not_bounce = list(unpacked[16])
                    session_cat = list(unpacked[17])

                    gru_preds = gru_model.predict_topk(
                        session_tokens,
                        session_tier,
                        session_pop,
                        session_p2p,
                        session_not_bounce,
                        session_cat,
                        session_dwell,
                        session_elapsed,
                        session_hour,
                        session_dow,
                        session_month,
                        session_gap,
                        session_age,
                        session_recency,
                        session_velocity,
                        session_decay,
                        k=6,
                        device=device,
                        banned=set(session_tokens),
                    )

                    gru_pids = [str(idx2pid.get(idx, idx)) for idx in gru_preds if idx > 0]
                    add_candidates(gru_pids)
                    if recs:
                        stats["gru"] += 1

            if len(product_seq) >= 3 and len(recs) < 6:
                before = len(recs)
                coocc_recs = coocc_recommend([str(x) for x in product_seq], coocc, top_k=6, lookback=5)
                add_candidates(coocc_recs)
                if len(recs) > before:
                    stats["coocc"] += 1

            if len(recs) < 6 and pd.notna(slug):
                if slug in cat2p and cat2p[slug]:
                    cat_recs = [str(k) for k, v in cat2p[slug].most_common(10)]
                    add_candidates(cat_recs)
                if len(recs) < 6:
                    query = slug_to_cat_map.get(slug, slug)
                    search_results = search_products(query, search_index, n=10)
                    add_candidates([str(s) for s in search_results])
                if not product_seq and recs:
                    stats["search"] += 1

        if not recs:
            stats["global"] += 1
        if len(recs) < 6:
            add_candidates([x for x in global_top])

        preds.append({"visit_id": vid, "product_ids": " ".join(recs[:6])})

    sub = pd.DataFrame(preds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    total = len(sub)
    print(f"GRU: {stats['gru']:,} ({stats['gru'] / total * 100:.1f}%)")
    print(f"Coocc: {stats['coocc']:,} ({stats['coocc'] / total * 100:.1f}%)")
    print(f"Search/cat2p: {stats['search']:,} ({stats['search'] / total * 100:.1f}%)")
    print(f"Global fallback: {stats['global']:,} ({stats['global'] / total * 100:.1f}%)")
    return output_path


def compare_outputs(candidate_path, reference_path):
    if not reference_path.exists():
        print(f"Reference file not found, skipping byte compare: {reference_path}")
        return
    matches = candidate_path.read_bytes() == reference_path.read_bytes()
    print(f"Byte-for-byte match with {reference_path}: {matches}")


def main():
    parser = argparse.ArgumentParser(description="Generate submission from saved inference artifacts.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE_OUTPUT)
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()

    output_path = generate_submission(args.output)
    if not args.no_compare:
        compare_outputs(output_path, args.reference)


if __name__ == "__main__":
    main()
