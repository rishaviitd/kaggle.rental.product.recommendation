# Baseline with merge_asof time-based session matching + Temporal GRU
import ast
import csv
from datetime import datetime
import json
import os
import pickle
import random
import re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

PAD_IDX = 0

# ==============================================================================
# SEED
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(100)

DATA_DIR = "data"
ARTIFACTS_DIR = Path("artifacts")

W_NEW = 2.0
SESSION_TOLERANCE = pd.Timedelta('2 hours')  # Configurable tolerance

# CUTOFF DATE: Last 6 Months (still used for co-occurrence, category maps, popularity)
# Max date is 2025-10-14, so we subtract 180 days
CUTOFF_DATE = pd.Timestamp('2025-10-14') - pd.Timedelta(days=180)
print(f"   📅 Recency Cutoff: {CUTOFF_DATE} (Last 6 Months)")

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
USE_SAMPLE_WEIGHTING = True
USE_PRICE_TIER = True
SAMPLE_WEIGHT_DECAY_RATE = 0.0004

print(f"🧪 Training Configuration:")
print(f"   Sample Weighting: {USE_SAMPLE_WEIGHTING}")
print(f"   Price Tier: {USE_PRICE_TIER}")
print(f"   Sample Weight Decay Rate: {SAMPLE_WEIGHT_DECAY_RATE}")

def save_pickle_artifact(name, value):
    path = ARTIFACTS_DIR / name
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_json_artifact(name, value):
    path = ARTIFACTS_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)

def save_inference_artifacts():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    pickle_artifacts = {
        "old_to_new.pkl": old_to_new,
        "slug_map.pkl": slug_map,
        "pid_to_cat.pkl": pid_to_cat,
        "cat2idx.pkl": cat2idx,
        "pid_to_cat_idx.pkl": pid_to_cat_idx,
        "pid_to_tier.pkl": pid_to_tier,
        "pid2idx.pkl": pid2idx,
        "idx2pid.pkl": idx2pid,
        "p2p.pkl": dict(p2p),
        "coocc.pkl": dict(coocc),
        "trigrams_dict.pkl": dict(trigrams_dict),
        "cat2p.pkl": dict(cat2p),
        "order_cooccur.pkl": dict(order_cooccur),
        "search_index.pkl": dict(search_index),
        "slug_to_cat_map.pkl": SLUG_TO_CAT_MAP,
    }

    for name, value in pickle_artifacts.items():
        save_pickle_artifact(name, value)

    config = {
        "pad_idx": PAD_IDX,
        "data_dir": DATA_DIR,
        "model_path": str(ARTIFACTS_DIR / "model.pt"),
        "w_new": W_NEW,
        "session_tolerance": str(SESSION_TOLERANCE),
        "cutoff_date": str(CUTOFF_DATE),
        "use_sample_weighting": USE_SAMPLE_WEIGHTING,
        "use_price_tier": USE_PRICE_TIER,
        "sample_weight_decay_rate": SAMPLE_WEIGHT_DECAY_RATE,
        "num_items": num_items,
        "num_categories": num_categories,
    }
    metadata = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "artifact_version": 1,
        "model_class": "GRURecDual",
        "inference_script": "inference.py",
    }

    save_json_artifact("global_top.json", global_top)
    save_json_artifact("config.json", config)
    save_json_artifact("metadata.json", metadata)

    saved_count = len(pickle_artifacts) + 3
    print(f"   ✅ Saved {saved_count} inference artifacts to: {ARTIFACTS_DIR}")

# ==============================================================================
# NEW: TIME-BASED SESSION MATCHING FUNCTIONS (merge_asof approach)
# ==============================================================================

def build_sessions_merge_asof(hits_path, visits_path, slug_map, tolerance=SESSION_TOLERANCE):
    """
    Build sessions by matching hits to visits using time-based proximity (merge_asof).
    This recovers more data than watch_id matching which has ~46% data loss.
    """
    print(f"   Loading hits from {hits_path}...")
    hits = pd.read_parquet(hits_path)
    hits['date_time'] = pd.to_datetime(hits['date_time'], format='ISO8601', errors='coerce')
    hits = hits.dropna(subset=['date_time', 'client_id'])
    print(f"   Loaded {len(hits):,} hits")
    
    print(f"   Loading visits from {visits_path}...")
    visits = pd.read_parquet(visits_path)
    visits['date_time'] = pd.to_datetime(visits['date_time'], format='ISO8601', errors='coerce')
    visits = visits.dropna(subset=['date_time', 'client_id', 'visit_id'])
    print(f"   Loaded {len(visits):,} visits")
    
    # Rename visit date_time to avoid confusion
    visits = visits.rename(columns={'date_time': 'visit_start'})
    
    # Sort both by client_id and time for merge_asof
    # Reset index after sorting to ensure proper merge
    hits_sorted = hits.sort_values(['client_id', 'date_time']).reset_index(drop=True)
    visits_sorted = visits.sort_values(['client_id', 'visit_start']).reset_index(drop=True)
    
    print(f"   Performing merge_asof with {tolerance} tolerance...")
    merged = pd.merge_asof(
        hits_sorted.sort_values('date_time'),  # Sort by time for merge_asof
        visits_sorted.sort_values('visit_start')[['client_id', 'visit_start', 'visit_id', 'project_id']],
        by='client_id',
        left_on='date_time',
        right_on='visit_start',
        direction='backward',
        tolerance=tolerance
    )
    
    # Count matching stats
    matched = merged['visit_id'].notna().sum()
    total = len(merged)
    print(f"   ✅ Matched {matched:,}/{total:,} hits ({matched/total*100:.2f}%)")
    
    return merged

def extract_product_sequences_from_merged(merged_df, slug_map):
    """Extract product sequences from merged dataframe."""
    # Filter to PRODUCT pages only and map slugs to product IDs
    products = merged_df[merged_df['page_type'] == 'PRODUCT'].copy()
    products['product_id'] = products['slug'].map(slug_map)
    products = products.dropna(subset=['visit_id', 'product_id'])
    products['product_id'] = products['product_id'].astype(int).astype(str)
    
    # Group by visit_id, sort by time, extract product sequence
    sequences = {}
    for visit_id, group in products.groupby('visit_id'):
        sorted_group = group.sort_values('date_time')
        seq = sorted_group['product_id'].tolist()
        if seq:
            sequences[str(visit_id)] = seq
    
    return sequences

def extract_actions_from_merged(merged_df, slug_map):
    """Extract (type, value) action sequences from merged dataframe for GRU training."""
    # Filter to PRODUCT and CATEGORY pages
    relevant = merged_df[merged_df['page_type'].isin(['PRODUCT', 'CATEGORY'])].copy()
    relevant = relevant.dropna(subset=['visit_id', 'slug'])
    
    # Map products
    relevant['product_id'] = relevant['slug'].map(slug_map)
    
    results = []
    for visit_id, group in relevant.groupby('visit_id'):
        sorted_group = group.sort_values('date_time')
        actions = []
        for _, row in sorted_group.iterrows():
            if row['page_type'] == 'PRODUCT' and pd.notna(row['product_id']):
                actions.append(('product', int(row['product_id'])))
            elif row['page_type'] == 'CATEGORY':
                actions.append(('category', row['slug']))
        
        if actions:
            results.append({
                'session_id': str(visit_id),
                'user_actions': actions,
                'project_id': str(group['project_id'].iloc[0]) if 'project_id' in group.columns else '1',
                'timestamp': sorted_group['date_time'].iloc[0]
            })
    
    return pd.DataFrame(results)

def get_last_slug_from_merged(merged_df):
    """Get the last slug for each visit from merged dataframe."""
    # Get the last hit for each visit
    relevant = merged_df.dropna(subset=['visit_id', 'slug'])
    last_hits = relevant.sort_values('date_time').groupby('visit_id').last()
    return last_hits['slug'].to_dict()

def get_last_cat_from_merged(merged_df):
    """Get the last category slug for each visit."""
    cats = merged_df[merged_df['page_type'] == 'CATEGORY'].copy()
    cats = cats.dropna(subset=['visit_id', 'slug'])
    if len(cats) == 0:
        return {}
    last_cats = cats.sort_values('date_time').groupby('visit_id').last()
    return last_cats['slug'].to_dict()

# ==============================================================================
# HELPER FUNCTIONS (kept from original)
# ==============================================================================

def build_cooccurrence(sessions, window_size=10, session_weights=None):
    coocc = defaultdict(Counter)
    if session_weights is None:
        session_weights = [1.0] * len(sessions)
    for seq, sw in zip(sessions, session_weights):
        if len(seq) < 2:
            continue
        for i, item in enumerate(seq):
            start = max(0, i - window_size)
            end = min(len(seq), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                neighbor = seq[j]
                dist = abs(i - j)
                weight = (1.0 / dist) * float(sw)
                coocc[item][neighbor] += weight
    return coocc

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

def scored_recommend(product_seq, coocc, p2p, trigrams_dict, cat2p_dict, order_cooccur,
                     last_cat=None, top_k=6):
    """Multi-signal scored recommendation for 1-2 product sessions."""
    seen = set(str(x) for x in product_seq)
    cand = defaultdict(float)
    str_seq = [str(x) for x in product_seq]

    # 1. Co-occurrence (existing, window-based)
    for i, item in enumerate(reversed(str_seq[-5:])):
        pos_weight = 1.0 / (i + 1)
        for neighbor, w in coocc.get(item, {}).items():
            cand[neighbor] += pos_weight * w

    # 2. P2P: what follows the last item (directional)
    for k, v in p2p.get(str_seq[-1], Counter()).most_common(20):
        cand[k] += v * 2.0

    # 3. Trigrams: what follows the last 2 items (bigram context)
    if len(str_seq) >= 2:
        key = (str_seq[-2], str_seq[-1])
        for k, v in trigrams_dict.get(key, Counter()).most_common(20):
            cand[k] += v * 3.0

    # 4. Category context (if available)
    if last_cat and last_cat in cat2p_dict:
        for k, v in cat2p_dict[last_cat].most_common(20):
            cand[k] += v * 1.5

    # 5. Orders co-purchase
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

def deduplicate_consecutive(actions):
    if not actions:
        return []
    deduped = [actions[0]]
    for i in range(1, len(actions)):
        if actions[i][0] == actions[i-1][0] and actions[i][1] == actions[i-1][1]:
            continue
        deduped.append(actions[i])
    return deduped

def deduplicate_consecutive_products(seq):
    """Deduplicate consecutive product IDs (for co-occurrence training)."""
    if not seq:
        return []
    deduped = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            deduped.append(seq[i])
    return deduped

def count_product_views(actions):
    return sum(1 for typ, _ in actions if typ == "product")

# ==============================================================================
# GRU FEATURE HELPERS
# ==============================================================================

def deduplicate_consecutive_gru_actions(actions):
    """Deduplicate consecutive product actions for GRU training."""
    if not actions:
        return []
    deduped = [actions[0]]
    for i in range(1, len(actions)):
        if actions[i][1] != deduped[-1][1]:
            deduped.append(actions[i])
    return deduped

def extract_gru_actions(
    merged_df,
    slug_map,
    pid_to_tier,
    pid_to_cat_idx,
):
    """Extract compact product sequences for the GRU: product, price tier, category."""
    relevant = merged_df[merged_df['page_type'] == 'PRODUCT'].copy()
    relevant = relevant.dropna(subset=['visit_id', 'slug'])
    relevant['product_id'] = relevant['slug'].map(slug_map)
    relevant = relevant.dropna(subset=['product_id'])

    results = []
    for visit_id, group in relevant.groupby('visit_id'):
        sorted_group = group.sort_values('date_time').reset_index(drop=True)
        actions = []
        pids = sorted_group['product_id'].tolist()

        for pid in pids:
            pid = int(pid)
            tier_idx = pid_to_tier.get(pid, 0)
            cat_idx = pid_to_cat_idx.get(pid, 0)
            actions.append(('product', pid, tier_idx, cat_idx))

        actions = deduplicate_consecutive_gru_actions(actions)

        if actions:
            results.append({
                'session_id': str(visit_id),
                'user_actions': actions,
                'timestamp': sorted_group['date_time'].iloc[0]
            })

    return pd.DataFrame(results)

def encode_sequence_for_gru(actions, pid2idx):
    """Encode product actions as token, price tier, and category indices."""
    result = []
    for typ, pid, tier_idx, cat_idx in actions:
        token_id = pid2idx.get(pid, 0)
        if token_id > 0:
            result.append((token_id, int(tier_idx), int(cat_idx)))
    return result

def encode_with_age(actions, session_ts, pid2idx, max_date):
    """Return (encoded_seq, days_old)."""
    enc_seq = encode_sequence_for_gru(actions, pid2idx)
    days_old = (max_date - session_ts).days
    return (enc_seq, days_old)

def summarize_action_features(df, label):
    """Lightweight feature coverage summary for debugging."""
    total = 0
    tier_nonzero = 0
    cat_nonzero = 0
    for actions in df['user_actions']:
        for a in actions:
            total += 1
            if a[2] > 0:
                tier_nonzero += 1
            if a[3] > 0:
                cat_nonzero += 1
    if total == 0:
        print(f"   {label} feature coverage -- no actions")
        return
    print(
        f"   {label} feature coverage -- tier:{tier_nonzero/total:.1%} "
        f"cat:{cat_nonzero/total:.1%}"
    )

print("🚀 STARTING PIPELINE WITH MERGE_ASOF + TEMPORAL GRU...")

# ==============================================================================
# 1. LOAD & MAP PRODUCTS
# ==============================================================================
print("\n[1/8] Loading & Mapping Products...")
map_df = pd.read_parquet(f'{DATA_DIR}/old_site_new_site_products.parquet')
old_to_new = dict(zip(map_df['old_site_id'], map_df['new_site_id']))

# Primary mapping source: products_all.csv
products_all = pd.read_parquet(f'{DATA_DIR}/products_all.parquet')
products_all = products_all.dropna(subset=['new_product_id'])
total_products_all = len(products_all)

def parse_slug_list(value):
    if pd.isna(value) or value == '':
        return []
    if isinstance(value, list):
        return [str(v) for v in value if pd.notna(v)]
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if pd.notna(v)]
    except (ValueError, SyntaxError):
        return []
    return []

tier_map = {f"Tier_{i}": i for i in range(1, 6)}
slug_map = {}
pid_to_cat = {}
pid_to_tier = {}
cat_set = set()

for _, row in products_all.iterrows():
    try:
        pid = int(row['new_product_id'])
    except (TypeError, ValueError):
        continue

    new_slug = row.get('new_slug')
    if pd.notna(new_slug) and str(new_slug).strip():
        slug_map[str(new_slug)] = pid

    for old_slug in parse_slug_list(row.get('old_slugs')):
        if old_slug and str(old_slug).strip():
            slug_map[str(old_slug)] = pid

    cat = row.get('main_category')
    if pd.notna(cat) and str(cat).strip():
        cat = str(cat)
        pid_to_cat[pid] = cat
        cat_set.add(cat)

    tier_raw = row.get('price_tier')
    if pd.notna(tier_raw) and str(tier_raw).strip():
        pid_to_tier[pid] = tier_map.get(str(tier_raw).strip(), 0)

cat2idx = {cat: i + 1 for i, cat in enumerate(sorted(cat_set))}
pid_to_cat_idx = {pid: cat2idx.get(cat, 0) for pid, cat in pid_to_cat.items()}
num_categories = len(cat2idx)

# Fallback slug coverage from legacy files (do not override products_all mappings)
old_prods = pd.read_parquet(f'{DATA_DIR}/old_site_products.parquet', columns=['id', 'slug'])
old_prods['new_id'] = old_prods['id'].map(old_to_new)
old_prods = old_prods.dropna(subset=['new_id', 'slug'])
for slug, pid in zip(old_prods['slug'], old_prods['new_id'].astype(int)):
    slug_map.setdefault(slug, pid)

new_prods = pd.read_parquet(f'{DATA_DIR}/new_site_products.parquet')
new_prods = new_prods.dropna(subset=['slug', 'id'])
for slug, pid in zip(new_prods['slug'], new_prods['id'].astype(int)):
    slug_map.setdefault(slug, pid)

print(f"   Old-site products mapped: {len(old_prods):,}")
print(f"   New-site products: {len(new_prods):,}")
print(f"   Total slug map entries: {len(slug_map):,}")
print(f"   Products with categories: {len(pid_to_cat):,}")
print(f"   Products with price tiers: {len(pid_to_tier):,}")
print(f"   Unique categories: {num_categories}")
print(f"   products_all rows: {total_products_all:,}")

# ==============================================================================
# 2. LEARN CATEGORY MAPPINGS
# ==============================================================================
print("\n[2/8] Learning Category Mappings...")
SLUG_TO_CAT_MAP = {
    'kolyaski': 'Коляски', 'kolyaski-yoyo': 'Коляски YoYo',
    'avtokresla': 'Автокресла', 'igrushki': 'Игрушки',
    'kacheli-shezlongi': 'Электрокачели', 'videonyani': 'Видеоняни',
    'sportkompleksy': 'Спортивные комплексы'
}
slug_votes = defaultdict(Counter)
chunk_size = 500000
cols = ['client_id', 'project_id', 'date_time', 'slug', 'page_type']
for hits_path in [f'{DATA_DIR}/metrika_hits.parquet', f'{DATA_DIR}/metrika_hits_test.parquet']:
    for chunk in [pd.read_parquet(hits_path, columns=cols)]:
        chunk['date'] = pd.to_datetime(chunk['date_time'], errors='coerce')
        # FILTER: Use only recent data for learning category maps
        chunk = chunk[chunk['date'] >= CUTOFF_DATE]
        
        df = chunk.dropna(subset=['date', 'slug']).sort_values(['client_id', 'date'])
        sessions = df.groupby(['client_id', 'project_id', df['date'].dt.date])
        for (_, proj_id, _), group in sessions:
            try:
                proj_id_int = int(proj_id)
            except (TypeError, ValueError):
                proj_id_int = 1
            weight = W_NEW if int(proj_id) == 0 else 1.0
            actions = group[['page_type', 'slug']].values.tolist()
            last_cat_slug = None
            for ptype, slug in actions:
                if ptype == 'CATEGORY':
                    last_cat_slug = slug
                elif ptype == 'PRODUCT' and last_cat_slug:
                    pid = slug_map.get(slug)
                    if pid:
                        rus_cat = pid_to_cat.get(int(pid))
                        if rus_cat:
                            slug_votes[last_cat_slug][rus_cat] += weight
print("   Compiling learned maps...")
learned_count = 0
for slug, votes in slug_votes.items():
    winner, count = votes.most_common(1)[0]
    if count > 5:
        SLUG_TO_CAT_MAP[slug] = winner
        learned_count += 1
print(f"   ✅ Learned {learned_count} new category mappings.")
print(f"   Total category mappings: {len(SLUG_TO_CAT_MAP)} (7 seed + {learned_count} learned)")

# ==============================================================================
# 3. BUILD SEARCH ENGINE
# ==============================================================================
print("\n[3/8] Building Search Engine...")
search_index = defaultdict(list)
for _, row in tqdm(new_prods.iterrows(), total=len(new_prods), desc="Indexing"):
    try:
        pid = int(row['id'])
        text = f"{row['name']} {row['brand']} {row['slug']} {row['main_category']}"
        text = text.lower().replace('-', ' ')
        tokens = set(re.split(r'[\s-]+', text))
        for t in tokens:
            if len(t) > 2:
                search_index[t].append(pid)
    except Exception:
        continue
print(f"   Index tokens: {len(search_index):,}")
print(f"   Products indexed: {len(new_prods):,}")

def search_products(query_slug, n=10):
    if pd.isna(query_slug):
        return []
    keywords = re.split(r'[\s-]+', str(query_slug).lower())
    keywords = [k for k in keywords if len(k) > 2]
    if not keywords:
        return []
    candidates = Counter()
    for k in keywords:
        if k in search_index:
            for pid in search_index[k]:
                candidates[pid] += 1
    return [pid for pid, _ in candidates.most_common(n)]

# ==============================================================================
# 4. BUILD GLOBAL POPULARITY
# ==============================================================================
print("\n[4/8] Building Global Popularity (Recent 6 Months)...")
global_cnt = Counter()
# Need 'date_time' to filter
cols_pop = cols 
for hits_path in [f'{DATA_DIR}/metrika_hits.parquet', f'{DATA_DIR}/metrika_hits_test.parquet']:
    for chunk in [pd.read_parquet(hits_path, columns=cols_pop)]:
        chunk['date'] = pd.to_datetime(chunk['date_time'], errors='coerce')
        # FILTER: Use only recent data for global popularity
        chunk = chunk[chunk['date'] >= CUTOFF_DATE]
        
        chunk['pid'] = chunk['slug'].map(slug_map)
        chunk['pid'] = pd.to_numeric(chunk['pid'], errors='coerce')
        df = chunk.dropna(subset=['pid'])
        df['pid'] = df['pid'].astype(int)
        for pid, proj_id in zip(df['pid'].tolist(), df['project_id'].tolist()):
            try:
                proj_id_int = int(proj_id)
            except (TypeError, ValueError):
                proj_id_int = 1
            weight = W_NEW if int(proj_id) == 0 else 1.0
            global_cnt[pid] += weight
global_top = [str(x) for x, _ in global_cnt.most_common(50)]
print(f"   Unique popular products: {len(global_cnt):,}")
print(f"   Top-50 fallback items loaded")

# ==============================================================================
# 5. BUILD TRAIN SESSIONS USING MERGE_ASOF (NEW APPROACH)
# ==============================================================================
print("\n[5/8] Building Train Sessions (merge_asof approach)...")

print("   Processing training data...")
train_merged = build_sessions_merge_asof(
    f'{DATA_DIR}/metrika_hits.parquet',
    f'{DATA_DIR}/metrika_visits.parquet',
    slug_map
)
train_sequences_by_visit = extract_product_sequences_from_merged(train_merged, slug_map)

print("   Processing test data...")
test_merged = build_sessions_merge_asof(
    f'{DATA_DIR}/metrika_hits_test.parquet',
    f'{DATA_DIR}/metrika_visits_test.parquet',
    slug_map
)
test_sequences_by_visit = extract_product_sequences_from_merged(test_merged, slug_map)

# Combine for training
train_sequences = []
train_session_weights = []
for vid, seq in train_sequences_by_visit.items():
    if len(seq) < 2:
        continue
    train_sequences.append(seq)
    train_session_weights.append(1.0)
for vid, seq in test_sequences_by_visit.items():
    if len(seq) < 2:
        continue
    train_sequences.append(seq)
    train_session_weights.append(1.0)
train_lens = [len(s) for s in train_sequences]
print(f"   Train sessions (product seqs): {len(train_sequences_by_visit):,}")
print(f"   Test sessions (product seqs): {len(test_sequences_by_visit):,}")
print(f"   Combined sessions (len>=2): {len(train_sequences):,}")
print(f"   Avg sequence length: {np.mean(train_lens):.1f} | Min: {min(train_lens)} | Max: {max(train_lens)}")

# ==============================================================================
# 6. BUILD CO-OCCURRENCE (Recent 6 Months)
# ==============================================================================
print("\n[6/8] Building Co-occurrence (Recent 6 Months)...")
# Filter train sessions to recent only
print(f"   Filtering co-occurrence data to >= {CUTOFF_DATE}...")
train_merged_recent = train_merged[train_merged['visit_start'] >= CUTOFF_DATE]
train_sequences_recent_map = extract_product_sequences_from_merged(train_merged_recent, slug_map)

# Use Recent Train + All Test (Test is always relevant)
coocc_sequences = []
coocc_weights = []

for vid, seq in train_sequences_recent_map.items():
    if len(seq) < 2: continue
    coocc_sequences.append(seq)
    coocc_weights.append(1.0)
    
for vid, seq in test_sequences_by_visit.items():
    if len(seq) < 2: continue
    coocc_sequences.append(seq)
    coocc_weights.append(1.0)

print(f"   Recent train sessions: {len(train_sequences_recent_map):,}")
print(f"   Test sessions: {len(test_sequences_by_visit):,}")
print(f"   Total co-occurrence sessions: {len(coocc_sequences):,}")
coocc = build_cooccurrence(coocc_sequences, window_size=10, session_weights=coocc_weights)
print(f"   Co-occurrence anchors (unique items): {len(coocc):,}")

# ==============================================================================
# 6.1 BUILD P2P + TRIGRAMS (last 6 months train + test)
# ==============================================================================
print("\n[6.1] Building P2P transitions and trigrams...")
p2p = defaultdict(Counter)
trigrams_dict = defaultdict(Counter)

for vid, seq in train_sequences_recent_map.items():
    for i in range(len(seq) - 1):
        p2p[seq[i]][seq[i+1]] += 1
        if i >= 1:
            trigrams_dict[(seq[i-1], seq[i])][seq[i+1]] += 1

for vid, seq in test_sequences_by_visit.items():
    for i in range(len(seq) - 1):
        p2p[seq[i]][seq[i+1]] += 1
        if i >= 1:
            trigrams_dict[(seq[i-1], seq[i])][seq[i+1]] += 1

total_p2p_transitions = sum(sum(c.values()) for c in p2p.values())
total_trigram_transitions = sum(sum(c.values()) for c in trigrams_dict.values())
print(f"   P2P anchors (unique items): {len(p2p):,}")
print(f"   P2P total transitions: {total_p2p_transitions:,}")
print(f"   Trigram keys (unique bigrams): {len(trigrams_dict):,}")
print(f"   Trigram total transitions: {total_trigram_transitions:,}")

# ==============================================================================
# 6.2 BUILD BEHAVIORAL CATEGORY-TO-PRODUCT (cat2p) (last 6 months train + test)
# ==============================================================================
print("\n[6.2] Building behavioral category-to-product (cat2p)...")

def build_cat2p_from_merged(merged_df, slug_map):
    cat2p_local = defaultdict(Counter)
    relevant = merged_df[merged_df['page_type'].isin(['PRODUCT', 'CATEGORY'])].copy()
    relevant = relevant.dropna(subset=['visit_id', 'slug'])
    relevant['product_id'] = relevant['slug'].map(slug_map)
    for visit_id, group in relevant.groupby('visit_id'):
        sorted_group = group.sort_values('date_time')
        last_cat = None
        for _, row in sorted_group.iterrows():
            if row['page_type'] == 'CATEGORY':
                last_cat = row['slug']
            elif row['page_type'] == 'PRODUCT' and last_cat is not None and pd.notna(row['product_id']):
                cat2p_local[last_cat][str(int(row['product_id']))] += 1
    return cat2p_local

cat2p = build_cat2p_from_merged(train_merged_recent, slug_map)
# Merge in test cat2p
cat2p_test = build_cat2p_from_merged(test_merged, slug_map)
for slug, counts in cat2p_test.items():
    cat2p[slug] += counts

total_cat2p_transitions = sum(sum(c.values()) for c in cat2p.values())
print(f"   cat2p categories: {len(cat2p):,}")
print(f"   cat2p total transitions: {total_cat2p_transitions:,}")

# ==============================================================================
# 6.3 BUILD ORDER CO-OCCURRENCE (all data)
# ==============================================================================
print("\n[6.3] Building order co-occurrence from order data...")
valid_pids = set(str(int(x)) for x in new_prods['id'].astype(int).unique())
order_cooccur = defaultdict(Counter)

# New site orders
try:
    new_orders = pd.read_parquet(f'{DATA_DIR}/new_site_orders.parquet')
    for oid, g in new_orders.groupby('id'):
        pids = list(set(str(int(p)) for p in g['product_id'].unique() if str(int(p)) in valid_pids))
        for i, p1 in enumerate(pids):
            for p2 in pids[i+1:]:
                order_cooccur[p1][p2] += 1
                order_cooccur[p2][p1] += 1
    print(f"   new_site_orders loaded: {len(new_orders):,} rows, {new_orders['id'].nunique():,} orders")
except Exception as e:
    print(f"   new_site_orders not available: {e}")

# Old site orders (map to new IDs)
try:
    old_orders = pd.read_parquet(f'{DATA_DIR}/old_site_orders.parquet')
    for oid, g in old_orders.groupby('id'):
        pids = []
        for p in g['product_id'].unique():
            new_id = old_to_new.get(int(p))
            if new_id and str(int(new_id)) in valid_pids:
                pids.append(str(int(new_id)))
        pids = list(set(pids))
        for i, p1 in enumerate(pids):
            for p2 in pids[i+1:]:
                order_cooccur[p1][p2] += 1
                order_cooccur[p2][p1] += 1
    print(f"   old_site_orders loaded: {len(old_orders):,} rows, {old_orders['id'].nunique():,} orders")
except Exception as e:
    print(f"   old_site_orders not available: {e}")

total_order_pairs = sum(sum(c.values()) for c in order_cooccur.values()) // 2
print(f"   Order co-occurrence anchors: {len(order_cooccur):,}")
print(f"   Order co-occurrence pairs: {total_order_pairs:,}")

# ==============================================================================
# 6.5 GRU TRAINING DATA
# ==============================================================================
print("\n[6.5/8] Building GRU Training Data...")

print("   Extracting train GRU actions...")
train_actions = extract_gru_actions(
    train_merged,
    slug_map,
    pid_to_tier,
    pid_to_cat_idx,
)

print("   Extracting test GRU actions...")
test_actions = extract_gru_actions(
    test_merged,
    slug_map,
    pid_to_tier,
    pid_to_cat_idx,
)
print(f"   Train GRU sessions: {len(train_actions):,}")
print(f"   Test GRU sessions: {len(test_actions):,}")
summarize_action_features(train_actions, "Train")
summarize_action_features(test_actions, "Test")

# Combine train + test for training
all_actions = pd.concat([train_actions, test_actions], ignore_index=True)
print(f"   Combined sessions: {len(all_actions):,}")

# Count products per session
def count_products_temporal(actions):
    return len(actions)  # All actions are products now

all_actions['product_count'] = all_actions['user_actions'].apply(count_products_temporal)

# Filter: Train on >=2 products
gru_train_data = all_actions[all_actions['product_count'] >= 2].copy()
lengths = gru_train_data['product_count'].values
print(f"   GRU train sessions (>=2 products): {len(gru_train_data):,}")
print(f"   Session length -- Mean: {np.mean(lengths):.1f} | Median: {np.median(lengths):.0f} | Max: {max(lengths)}")

# Build vocabulary (product-only, no categories)
print("\n   Building vocabulary...")
all_products = set()
for actions in gru_train_data['user_actions']:
    for typ, pid, *_ in actions:
        all_products.add(pid)

pid2idx = {pid: i+1 for i, pid in enumerate(sorted(all_products))}
idx2pid = {i: pid for pid, i in pid2idx.items()}
num_items = len(pid2idx)
print(f"   Products: {num_items}")

# Encode sequences + session age for sample weighting
print("\nEncoding GRU sequences...")
max_date = train_merged['visit_start'].max()
gru_train_data['timestamp'] = pd.to_datetime(gru_train_data['timestamp'])

gru_sequences = []
for _, row in gru_train_data.iterrows():
    seq_data = encode_with_age(
        row['user_actions'],
        row['timestamp'],
        pid2idx,
        max_date,
    )
    if len(seq_data[0]) >= 2:
        gru_sequences.append(seq_data)

ages = [s[1] for s in gru_sequences]
print(f"   Encoded sequences: {len(gru_sequences):,}")
print(f"   Session age -- Mean: {np.mean(ages):.0f} days | Min: {min(ages)} | Max: {max(ages)}")

# ==============================================================================
# TEMPORAL GRU MODEL + TRAINING
# ==============================================================================
print("\nTraining Temporal GRU Model...")

class TemporalSessionDataset(Dataset):
    """Dataset of compact GRU sessions and session age."""
    def __init__(self, sessions, min_len=2):
        self.sessions = [s for s in sessions if len(s[0]) >= min_len]

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, i):
        return self.sessions[i]

def collate_temporal_sequences(batch):
    """Collate compact GRU sequences with sample weights."""
    valid_batch = [s for s in batch if len(s[0]) >= 2]
    if not valid_batch:
        return (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1,), dtype=torch.float)
        )

    max_len = max(len(s[0]) for s in valid_batch)
    B = len(valid_batch)
    T = max_len - 1

    x = torch.full((B, T), PAD_IDX, dtype=torch.long)
    tier = torch.zeros((B, T), dtype=torch.long)
    cat = torch.zeros((B, T), dtype=torch.long)
    y = torch.full((B, T), PAD_IDX, dtype=torch.long)
    sample_weights = torch.ones((B,), dtype=torch.float)

    for i, (seq, days_old) in enumerate(valid_batch):
        L = len(seq)

        # Calculate sample weight
        if USE_SAMPLE_WEIGHTING:
            weight = np.exp(-SAMPLE_WEIGHT_DECAY_RATE * days_old)
            sample_weights[i] = float(weight)

        for t in range(L - 1):
            x[i, t] = seq[t][0]       # token_id
            tier[i, t] = seq[t][1]    # price tier idx
            cat[i, t] = seq[t][2]     # category idx
            y[i, t] = seq[t + 1][0]     # next token_id

    return x, tier, cat, y, sample_weights

class GRURecDual(nn.Module):
    """Dual-path GRU: item path + category path."""
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
        use_tier=True,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_categories = num_categories
        self.use_tier = use_tier

        self.item_emb = nn.Embedding(num_items + 1, item_emb_dim, padding_idx=PAD_IDX)
        nn.init.xavier_uniform_(self.item_emb.weight.data)

        # Price tier embedding (Tier_1..Tier_5 + padding)
        if self.use_tier:
            self.tier_emb = nn.Embedding(6, tier_emb_dim, padding_idx=PAD_IDX)

        # Category embedding for dual path
        self.cat_emb = nn.Embedding(num_categories + 1, cat_emb_dim, padding_idx=PAD_IDX)

        in_dim = item_emb_dim + (tier_emb_dim if self.use_tier else 0)

        self.item_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.item_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.cat_proj = nn.Sequential(
            nn.Linear(cat_emb_dim, cat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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
        cat=None,
    ):
        """
        x: (B, T) token indices
        tier: (B, T) price tier indices
        cat: (B, T) category indices
        """
        e = self.item_emb(x)  # (B, T, E)
        feats_list = [e]
        if self.use_tier:
            t = self.tier_emb(tier) if tier is not None else torch.zeros_like(e[..., :0])
            feats_list.append(t)

        feats = torch.cat(feats_list, dim=-1)
        item_h = self.item_proj(feats)
        item_h, _ = self.item_gru(item_h)

        if cat is None:
            cat = torch.zeros_like(x)
        cat_e = self.cat_emb(cat)
        cat_h = self.cat_proj(cat_e)
        cat_h, _ = self.cat_gru(cat_h)

        fused = torch.cat([item_h, cat_h], dim=-1)
        return self.out(fused)

    @torch.no_grad()
    def predict_topk(
        self,
        session_tokens,
        session_tier,
        session_cat,
        k=6,
        device='cpu',
        banned=None,
    ):
        """Predict top-k next items with compact product and category features."""
        if not session_tokens:
            return []

        x = torch.tensor(session_tokens, dtype=torch.long, device=device).unsqueeze(0)
        tier_t = torch.tensor(session_tier, dtype=torch.long, device=device).unsqueeze(0)
        cat_t = torch.tensor(session_cat, dtype=torch.long, device=device).unsqueeze(0)

        logits = self.forward(
            x,
            tier_t,
            cat_t,
        )[0, -1]  # (V,)

        if banned:
            for b in banned:
                if 0 < b < len(logits):
                    logits[b] = -float('inf')

        logits[0] = -float('inf')  # Mask padding

        _, top_indices = torch.topk(logits, min(k + len(session_tokens), len(logits)))

        result = []
        for idx in top_indices.tolist():
            if idx > 0 and idx not in session_tokens:
                result.append(idx)
                if len(result) >= k:
                    break

        return result

# Initialize dual-path model
gru_model = GRURecDual(
    num_items=num_items,
    num_categories=num_categories,
    item_emb_dim=128,
    tier_emb_dim=4,
    cat_emb_dim=8,
    hidden_dim=128,
    cat_hidden_dim=96,
    num_layers=1,
    dropout=0.2,
    use_tier=USE_PRICE_TIER,
)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
gru_model.to(device)
print(f"   Device: {device}")
print(f"   Parameters: {sum(p.numel() for p in gru_model.parameters()):,}")
print(f"   Sample Weighting: {'Enabled' if USE_SAMPLE_WEIGHTING else 'Disabled'}")
print(f"   GRU item input dim: {gru_model.item_proj[0].in_features}")

# Training
train_ds = TemporalSessionDataset(gru_sequences, min_len=2)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_temporal_sequences)
print(f"   Training samples: {len(train_ds):,}")
print(f"   Batches per epoch: {len(train_loader):,}")

opt = torch.optim.AdamW(gru_model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

def weighted_ce_loss(logits, targets, weights, pad_idx=0):
    """Calculates standard CrossEntropyLoss with sample weighting."""
    mask = targets != pad_idx
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits_flat = logits[mask]
    targets_flat = targets[mask]

    # Expand weights: (B,) -> (B, T) -> Masked -> (N,)
    B, T = mask.shape
    weights_expanded = weights.unsqueeze(1).expand(B, T)
    weights_flat = weights_expanded[mask]

    # Standard CE with reduction='none' to apply weights manually
    ce_loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none')

    # Weighted mean
    return (ce_loss * weights_flat).sum() / weights_flat.sum()

EPOCHS = 3
best_loss = float('inf')
model_path = 'artifacts/model.pt'

for epoch in range(EPOCHS):
    gru_model.train()
    total_loss, n_batches = 0.0, 0

    for x, tier_b, cat_b, y, w in train_loader:
        x = x.to(device)
        tier_b = tier_b.to(device)
        cat_b = cat_b.to(device)
        y = y.to(device)
        w = w.to(device)

        opt.zero_grad()
        logits = gru_model(
            x,
            tier_b,
            cat_b,
        )
        loss = weighted_ce_loss(logits, y, w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gru_model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(gru_model.state_dict(), model_path)

    print(f"   Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")

gru_model.load_state_dict(torch.load(model_path))
gru_model.eval()
print(f"   ✅ Temporal GRU Training Complete! Final best loss: {best_loss:.4f}")

# ==============================================================================
# 7. SAVE INFERENCE ARTIFACTS
# ==============================================================================
print("\n[7/7] Saving inference artifacts...")
save_inference_artifacts()
print("\nTraining complete. Run `uv run inference.py` to generate the submission file.")
