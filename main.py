# Baseline with merge_asof time-based session matching + Temporal GRU
import ast
import csv
from datetime import datetime
import json
import os
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
SUBMISSION_FILE = 'output/submission.csv'

W_NEW = 2.0
SESSION_TOLERANCE = pd.Timedelta('2 hours')  # Configurable tolerance

# CUTOFF DATE: Last 6 Months (still used for co-occurrence, category maps, popularity)
# Max date is 2025-10-14, so we subtract 180 days
CUTOFF_DATE = pd.Timestamp('2025-10-14') - pd.Timedelta(days=180)
print(f"   📅 Recency Cutoff: {CUTOFF_DATE} (Last 6 Months)")

# ==============================================================================
# TEMPORAL FEATURE CONFIGURATION
# ==============================================================================
USE_DWELL_TIME = True
USE_SESSION_ELAPSED = True
USE_TIME_OF_DAY = True
USE_DAY_OF_WEEK = True
USE_TIME_DECAY = True
USE_SAMPLE_WEIGHTING = True
USE_MONTH = True
USE_INTER_SESSION_GAP = True
USE_PRODUCT_AGE = True
USE_PRODUCT_RECENCY = True
USE_PRODUCT_VELOCITY = True
MAX_DWELL_SECONDS = 300   # Cap dwell time at 5 minutes
MAX_SESSION_SECONDS = 3600  # Cap session elapsed at 1 hour
MAX_GAP_DAYS = 365 * 2
MAX_PRODUCT_AGE_DAYS = 365 * 4
MAX_PRODUCT_RECENCY_DAYS = 365 * 4
MAX_PRODUCT_VELOCITY = 10.0

print(f"🧪 Temporal Features Configuration:")
print(f"   Dwell Time: {USE_DWELL_TIME}")
print(f"   Session Elapsed: {USE_SESSION_ELAPSED}")
print(f"   Time of Day: {USE_TIME_OF_DAY}")
print(f"   Day of Week: {USE_DAY_OF_WEEK}")
print(f"   Time Decay: {USE_TIME_DECAY}")
print(f"   Sample Weighting: {USE_SAMPLE_WEIGHTING}")
print(f"   Month (Seasonality): {USE_MONTH}")
print(f"   Inter-session Gap: {USE_INTER_SESSION_GAP}")
print(f"   Product Age: {USE_PRODUCT_AGE}")
print(f"   Product Recency: {USE_PRODUCT_RECENCY}")
print(f"   Product Velocity: {USE_PRODUCT_VELOCITY}")

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
# NEW: TIME FEATURE HELPERS
# ==============================================================================

def log_norm(value, max_value):
    if max_value <= 0:
        return 0.0
    value = max(0.0, min(float(value), float(max_value)))
    return np.log1p(value) / np.log1p(max_value)


def compute_product_time_stats(hits_paths, slug_map, chunk_size=500000):
    daily_counts = Counter()
    first_seen = {}
    last_seen = {}
    global_max_date = None

    cols_time = ['date_time', 'slug', 'page_type']
    for hits_path in hits_paths:
        for chunk in [pd.read_parquet(hits_path, columns=cols_time)]:
            chunk['date'] = pd.to_datetime(chunk['date_time'], errors='coerce').dt.date
            chunk = chunk.dropna(subset=['date', 'slug'])
            chunk = chunk[chunk['page_type'] == 'PRODUCT']
            if chunk.empty:
                continue
            chunk['pid'] = chunk['slug'].map(slug_map)
            chunk = chunk.dropna(subset=['pid'])
            if chunk.empty:
                continue
            chunk['pid'] = chunk['pid'].astype(int)

            grouped = chunk.groupby(['date', 'pid']).size()
            for (d, pid), cnt in grouped.items():
                daily_counts[(d, pid)] += int(cnt)

            per_pid_dates = chunk.groupby('pid')['date'].agg(['min', 'max'])
            for pid, row in per_pid_dates.iterrows():
                dmin = row['min']
                dmax = row['max']
                if pid not in first_seen or dmin < first_seen[pid]:
                    first_seen[pid] = dmin
                if pid not in last_seen or dmax > last_seen[pid]:
                    last_seen[pid] = dmax

            max_date_chunk = chunk['date'].max()
            if global_max_date is None or max_date_chunk > global_max_date:
                global_max_date = max_date_chunk

    velocity_by_pid = {}
    velocity_mean = 1.0
    max_age_days = 0
    max_recency_days = 0

    if daily_counts:
        df = pd.DataFrame(
            [(d, pid, cnt) for (d, pid), cnt in daily_counts.items()],
            columns=['date', 'pid', 'cnt']
        ).sort_values('date')
        pivot = df.pivot_table(index='date', columns='pid', values='cnt', fill_value=0)
        roll3 = pivot.rolling(3, min_periods=1).mean()
        roll14 = pivot.rolling(14, min_periods=1).mean()
        velocity = roll3 / (roll14 + 1e-6)
        velocity = velocity.clip(upper=MAX_PRODUCT_VELOCITY)

        velocity_vals = velocity.to_numpy().flatten()
        velocity_vals = velocity_vals[~np.isnan(velocity_vals)]
        if velocity_vals.size > 0:
            velocity_mean = float(np.nanmean(velocity_vals))

        for pid in velocity.columns:
            velocity_by_pid[int(pid)] = velocity[pid].to_dict()

    if global_max_date is None:
        global_max_date = pd.Timestamp('1970-01-01').date()

    product_recency = {}
    for pid, dmin in first_seen.items():
        age_days = (global_max_date - dmin).days
        if age_days > max_age_days:
            max_age_days = age_days
    for pid, dmax in last_seen.items():
        recency_days = (global_max_date - dmax).days
        product_recency[pid] = recency_days
        if recency_days > max_recency_days:
            max_recency_days = recency_days

    return {
        'first_seen': first_seen,
        'last_seen': last_seen,
        'recency': product_recency,
        'velocity_by_pid': velocity_by_pid,
        'velocity_mean': velocity_mean,
        'global_max_date': global_max_date,
        'max_age_days': max_age_days,
        'max_recency_days': max_recency_days,
    }


def compute_visit_gap_map(visit_paths):
    visits_list = []
    for vp in visit_paths:
        df = pd.read_parquet(vp, columns=['visit_id', 'client_id', 'date_time'])
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        df = df.dropna(subset=['date_time', 'client_id', 'visit_id'])
        df = df.rename(columns={'date_time': 'visit_start'})
        visits_list.append(df)
    visits = pd.concat(visits_list, ignore_index=True)
    visits = visits.sort_values(['client_id', 'visit_start'])
    visits['prev_start'] = visits.groupby('client_id')['visit_start'].shift(1)
    visits['gap_days'] = (visits['visit_start'] - visits['prev_start']).dt.total_seconds() / 86400.0
    visits['gap_days'] = visits['gap_days'].fillna(0.0)
    return dict(zip(visits['visit_id'].astype(str), visits['gap_days'].astype(float)))
# ==============================================================================
# TEMPORAL HELPER FUNCTIONS (from baseline_gru_temporal.py)
# ==============================================================================

def deduplicate_consecutive_with_temporal(actions):
    """Deduplicate consecutive items and aggregate dwell times, keep first elapsed."""
    if not actions:
        return []
    # actions: [(type, pid, dwell, elapsed, hour, dow, month, gap, age, recency, velocity,
    #            tier_idx, pop_score, not_bounce, cat_idx), ...]
    deduped = [actions[0]]
    for i in range(1, len(actions)):
        (curr_type, curr_pid, curr_dwell, curr_elapsed, curr_hour, curr_dow,
         curr_month, curr_gap, curr_age, curr_recency, curr_velocity,
         curr_tier, curr_pop, curr_nb, curr_cat) = actions[i]
        (prev_type, prev_pid, prev_dwell, prev_elapsed, prev_hour, prev_dow,
         prev_month, prev_gap, prev_age, prev_recency, prev_velocity,
         prev_tier, prev_pop, prev_nb, prev_cat) = deduped[-1]
        if curr_type == prev_type and curr_pid == prev_pid:
            # Same item - aggregate dwell time, keep first elapsed/hour/dow
            deduped[-1] = (
                prev_type, prev_pid, prev_dwell + curr_dwell, prev_elapsed, prev_hour, prev_dow,
                prev_month, prev_gap, prev_age, prev_recency, prev_velocity,
                prev_tier, prev_pop, max(prev_nb, curr_nb), prev_cat
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
    """Extract action sequences with dwell time and session elapsed from merged dataframe."""
    relevant = merged_df[merged_df['page_type'] == 'PRODUCT'].copy()
    relevant = relevant.dropna(subset=['visit_id', 'slug'])
    relevant['product_id'] = relevant['slug'].map(slug_map)
    relevant = relevant.dropna(subset=['product_id'])

    results = []
    missing_first_seen = 0
    missing_recency = 0
    missing_velocity = 0
    for visit_id, group in relevant.groupby('visit_id'):
        sorted_group = group.sort_values('date_time').reset_index(drop=True)
        actions = []
        timestamps = sorted_group['date_time'].tolist()
        pids = sorted_group['product_id'].tolist()
        if 'not_bounce' in sorted_group.columns:
            not_bounce_vals = sorted_group['not_bounce'].fillna(0).astype(int).tolist()
        else:
            not_bounce_vals = [0] * len(pids)
        session_start = timestamps[0]
        gap_days = visit_gap_map.get(str(visit_id), 0.0) if USE_INTER_SESSION_GAP else 0.0

        for i in range(len(pids)):
            # Dwell time = time until next action (or 0 for last)
            if i < len(timestamps) - 1:
                dwell_seconds = (timestamps[i+1] - timestamps[i]).total_seconds()
                dwell_seconds = min(max(dwell_seconds, 0), MAX_DWELL_SECONDS)
            else:
                dwell_seconds = 0
            # Session elapsed = time since session start
            elapsed_seconds = (timestamps[i] - session_start).total_seconds()
            elapsed_seconds = min(max(elapsed_seconds, 0), MAX_SESSION_SECONDS)
            # Time of Day (Hour)
            hour = timestamps[i].hour
            # Day of Week (0=Monday, 6=Sunday)
            dow = timestamps[i].dayofweek
            # Month (1-12)
            month = timestamps[i].month

            pid = int(pids[i])
            action_date = timestamps[i].date()

            # Product age (days since first seen)
            first_seen_date = product_first_seen.get(pid)
            if first_seen_date is None:
                missing_first_seen += 1
                age_days = 0.0
            else:
                age_days = (action_date - first_seen_date).days
                age_days = max(age_days, 0.0)

            # Product recency (days since last seen, as of global max date)
            recency_days = product_recency.get(pid)
            if recency_days is None:
                missing_recency += 1
                recency_days = 0.0

            # Product velocity (short-term / mid-term)
            velocity = velocity_by_pid.get(pid, {}).get(action_date)
            if velocity is None:
                missing_velocity += 1
                velocity = velocity_mean

            tier_idx = pid_to_tier.get(pid, 0)
            pop_score = pid_to_pop.get(pid, 0.0)
            not_bounce = int(not_bounce_vals[i]) if i < len(not_bounce_vals) else 0
            cat_idx = pid_to_cat_idx.get(pid, 0)

            actions.append((
                'product', pid, dwell_seconds, elapsed_seconds, hour, dow, month,
                gap_days, age_days, recency_days, velocity,
                tier_idx, pop_score, not_bounce, cat_idx
            ))

        # Apply consecutive deduplication
        actions = deduplicate_consecutive_with_temporal(actions)

        if actions:
            results.append({
                'session_id': str(visit_id),
                'user_actions': actions,
                'timestamp': sorted_group['date_time'].iloc[0]
            })

    if missing_first_seen or missing_recency or missing_velocity:
        print(f"   Missing first_seen: {missing_first_seen:,} | recency: {missing_recency:,} | velocity: {missing_velocity:,}")

    return pd.DataFrame(results)

def encode_sequence_with_temporal(actions, pid2idx, p2p=None, p2p_totals=None):
    """Encode sequence with all temporal features + product metadata + transition strength."""
    result = []
    T = len(actions)
    for i, (typ, pid, dwell, elapsed, hour, dow, month, gap_days, age_days, recency_days, velocity,
            tier_idx, pop_score, not_bounce, cat_idx) in enumerate(actions):
        token_id = pid2idx.get(pid, 0)
        if token_id > 0:
            # 1. Normalize dwell time
            norm_dwell = np.log1p(dwell) / np.log1p(MAX_DWELL_SECONDS)
            # 2. Normalize elapsed time
            norm_elapsed = np.log1p(elapsed) / np.log1p(MAX_SESSION_SECONDS)
            # 3. Time of Day (Cyclical)
            hour_sin = np.sin(2 * np.pi * hour / 24.0)
            hour_cos = np.cos(2 * np.pi * hour / 24.0)
            # 4. Day of Week (Embedding index 0-6)
            dow_idx = dow
            # 5. Month (Cyclical)
            month_sin = np.sin(2 * np.pi * month / 12.0) if USE_MONTH else 0.0
            month_cos = np.cos(2 * np.pi * month / 12.0) if USE_MONTH else 0.0
            # 6. Inter-session gap
            gap_norm = log_norm(gap_days, MAX_GAP_DAYS) if USE_INTER_SESSION_GAP else 0.0
            # 7. Product age
            age_norm = log_norm(age_days, MAX_PRODUCT_AGE_DAYS) if USE_PRODUCT_AGE else 0.0
            # 8. Product recency
            recency_norm = log_norm(recency_days, MAX_PRODUCT_RECENCY_DAYS) if USE_PRODUCT_RECENCY else 0.0
            # 9. Product velocity
            vel_norm = log_norm(velocity, MAX_PRODUCT_VELOCITY) if USE_PRODUCT_VELOCITY else 0.0
            # 10. Time Decay (Positional) -- exp(-decay * distance_from_end)
            decay = np.exp(-0.1 * (T - 1 - i))

            # 11. P2P transition strength from previous item (normalized)
            p2p_score = 0.0
            if p2p is not None and p2p_totals is not None and i > 0:
                prev_pid = actions[i - 1][1]
                prev_key = str(prev_pid)
                curr_key = str(pid)
                count = p2p.get(prev_key, {}).get(curr_key, 0)
                total = p2p_totals.get(prev_key, 0)
                if total > 0:
                    p2p_score = float(count) / float(total)

            result.append((
                token_id, norm_dwell, norm_elapsed, hour_sin, hour_cos, dow_idx,
                month_sin, month_cos, gap_norm, age_norm, recency_norm, vel_norm, decay,
                int(tier_idx), float(pop_score), float(p2p_score), float(not_bounce), int(cat_idx)
            ))
    return result

def encode_with_age(actions, session_ts, pid2idx, max_date, p2p=None, p2p_totals=None):
    """Return (encoded_seq, days_old)."""
    enc_seq = encode_sequence_with_temporal(actions, pid2idx, p2p=p2p, p2p_totals=p2p_totals)
    days_old = (max_date - session_ts).days
    return (enc_seq, days_old)

def summarize_action_features(df, label):
    """Lightweight feature coverage summary for debugging."""
    total = 0
    tier_nonzero = 0
    pop_nonzero = 0
    nb_nonzero = 0
    cat_nonzero = 0
    for actions in df['user_actions']:
        for a in actions:
            total += 1
            if a[11] > 0:
                tier_nonzero += 1
            if a[12] > 0:
                pop_nonzero += 1
            if a[13] > 0:
                nb_nonzero += 1
            if a[14] > 0:
                cat_nonzero += 1
    if total == 0:
        print(f"   {label} feature coverage -- no actions")
        return
    print(
        f"   {label} feature coverage -- tier:{tier_nonzero/total:.1%} "
        f"pop:{pop_nonzero/total:.1%} not_bounce:{nb_nonzero/total:.1%} "
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
# 1.1 BUILD PRODUCT TIME FEATURES (Train + Test Hits)
# ==============================================================================
print("\n[1.1/8] Building Product Time Features...")
time_stats = compute_product_time_stats(
    [f'{DATA_DIR}/metrika_hits.parquet', f'{DATA_DIR}/metrika_hits_test.parquet'],
    slug_map,
    chunk_size=500000
)
product_first_seen = time_stats['first_seen']
product_last_seen = time_stats['last_seen']
product_recency = time_stats['recency']
velocity_by_pid = time_stats['velocity_by_pid']
velocity_mean = time_stats['velocity_mean']
GLOBAL_MAX_DATE = time_stats['global_max_date']

if time_stats['max_age_days'] > 0:
    MAX_PRODUCT_AGE_DAYS = min(MAX_PRODUCT_AGE_DAYS, time_stats['max_age_days'])
if time_stats['max_recency_days'] > 0:
    MAX_PRODUCT_RECENCY_DAYS = min(MAX_PRODUCT_RECENCY_DAYS, time_stats['max_recency_days'])

print(f"   Products with first_seen: {len(product_first_seen):,}")
print(f"   Products with last_seen: {len(product_last_seen):,}")
print(f"   Velocity mean: {velocity_mean:.3f}")
print(f"   Max product age days (cap): {MAX_PRODUCT_AGE_DAYS}")
print(f"   Max product recency days (cap): {MAX_PRODUCT_RECENCY_DAYS}")

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

# Build per-product popularity score (log-normalized)
max_pop = max(global_cnt.values()) if global_cnt else 1.0
pid_to_pop = {int(pid): log_norm(cnt, max_pop) for pid, cnt in global_cnt.items()}
if pid_to_pop:
    pop_vals = list(pid_to_pop.values())
    print(f"   Pop score range: {min(pop_vals):.4f} - {max(pop_vals):.4f}")

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

# Precompute totals for normalized p2p transition strength
p2p_totals = {k: sum(v.values()) for k, v in p2p.items()}
if p2p_totals:
    totals = np.array(list(p2p_totals.values()), dtype=float)
    print(f"   P2P totals -- Mean: {totals.mean():.1f} | Max: {totals.max():.0f}")

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
# 6.5 GRU TRAINING DATA WITH TEMPORAL FEATURES
# ==============================================================================
print("\n[6.5/8] Building GRU Training Data with Temporal Features...")

print("   Computing inter-session gaps...")
visit_gap_map = compute_visit_gap_map(
    [f'{DATA_DIR}/metrika_visits.parquet', f'{DATA_DIR}/metrika_visits_test.parquet']
)
if visit_gap_map:
    gap_vals = np.array(list(visit_gap_map.values()), dtype=float)
    print(f"   Gap days -- Mean: {gap_vals.mean():.2f} | Median: {np.median(gap_vals):.2f} | Max: {gap_vals.max():.2f}")

print("   Extracting train actions with temporal features...")
train_actions = extract_actions_with_temporal(
    train_merged,
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

print("   Extracting test actions with temporal features...")
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
print(f"   Train temporal sessions: {len(train_actions):,}")
print(f"   Test temporal sessions: {len(test_actions):,}")
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

# Encode sequences with temporal features + session age for sample weighting
print("\nEncoding sequences with temporal features...")
max_date = train_merged['visit_start'].max()
gru_train_data['timestamp'] = pd.to_datetime(gru_train_data['timestamp'])

gru_sequences = []
for _, row in gru_train_data.iterrows():
    seq_data = encode_with_age(
        row['user_actions'],
        row['timestamp'],
        pid2idx,
        max_date,
        p2p=p2p,
        p2p_totals=p2p_totals
    )
    if len(seq_data[0]) >= 2:
        gru_sequences.append(seq_data)

ages = [s[1] for s in gru_sequences]
print(f"   Encoded sequences: {len(gru_sequences):,}")
print(f"   Session age -- Mean: {np.mean(ages):.0f} days | Min: {min(ages)} | Max: {max(ages)}")
if gru_sequences:
    p2p_vals = []
    for seq, _ in gru_sequences[:200]:
        for row in seq:
            p2p_vals.append(row[15])
    if p2p_vals:
        p2p_vals = np.array(p2p_vals, dtype=float)
        print(f"   P2P score sample -- Mean: {p2p_vals.mean():.4f} | Max: {p2p_vals.max():.4f}")

# ==============================================================================
# TEMPORAL GRU MODEL + TRAINING
# ==============================================================================
print("\nTraining Temporal GRU Model...")

class TemporalSessionDataset(Dataset):
    """Dataset of sessions with temporal features and session age."""
    def __init__(self, sessions, min_len=2):
        self.sessions = [s for s in sessions if len(s[0]) >= min_len]

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, i):
        return self.sessions[i]

def collate_temporal_sequences(batch):
    """Collate sequences with dwell time, elapsed time, and sample weights."""
    valid_batch = [s for s in batch if len(s[0]) >= 2]
    if not valid_batch:
        return (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1, 2), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1, 2), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.float),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1,), dtype=torch.float)
        )

    max_len = max(len(s[0]) for s in valid_batch)
    B = len(valid_batch)
    T = max_len - 1

    x = torch.full((B, T), PAD_IDX, dtype=torch.long)
    dwell = torch.zeros((B, T), dtype=torch.float)
    elapsed = torch.zeros((B, T), dtype=torch.float)
    hour = torch.zeros((B, T, 2), dtype=torch.float)
    dow = torch.zeros((B, T), dtype=torch.long)
    month = torch.zeros((B, T, 2), dtype=torch.float)
    gap = torch.zeros((B, T), dtype=torch.float)
    age = torch.zeros((B, T), dtype=torch.float)
    recency = torch.zeros((B, T), dtype=torch.float)
    velocity = torch.zeros((B, T), dtype=torch.float)
    decay = torch.zeros((B, T), dtype=torch.float)
    tier = torch.zeros((B, T), dtype=torch.long)
    pop = torch.zeros((B, T), dtype=torch.float)
    p2p_score = torch.zeros((B, T), dtype=torch.float)
    not_bounce = torch.zeros((B, T), dtype=torch.float)
    cat = torch.zeros((B, T), dtype=torch.long)
    y = torch.full((B, T), PAD_IDX, dtype=torch.long)
    sample_weights = torch.ones((B,), dtype=torch.float)

    # Sample weight decay rate
    decay_rate = 0.0004

    for i, (seq, days_old) in enumerate(valid_batch):
        L = len(seq)

        # Calculate sample weight
        if USE_SAMPLE_WEIGHTING:
            weight = np.exp(-decay_rate * days_old)
            sample_weights[i] = float(weight)

        for t in range(L - 1):
            x[i, t] = seq[t][0]       # token_id
            dwell[i, t] = seq[t][1]   # norm_dwell
            elapsed[i, t] = seq[t][2] # norm_elapsed
            hour[i, t, 0] = seq[t][3] # hour_sin
            hour[i, t, 1] = seq[t][4] # hour_cos
            dow[i, t] = seq[t][5]     # dow_idx
            month[i, t, 0] = seq[t][6]  # month_sin
            month[i, t, 1] = seq[t][7]  # month_cos
            gap[i, t] = seq[t][8]       # gap_norm
            age[i, t] = seq[t][9]       # age_norm
            recency[i, t] = seq[t][10]  # recency_norm
            velocity[i, t] = seq[t][11] # velocity_norm
            decay[i, t] = seq[t][12]    # time_decay
            tier[i, t] = seq[t][13]     # price tier idx
            pop[i, t] = seq[t][14]      # popularity score
            p2p_score[i, t] = seq[t][15]  # p2p transition strength
            not_bounce[i, t] = seq[t][16] # not_bounce flag
            cat[i, t] = seq[t][17]      # category idx
            y[i, t] = seq[t + 1][0]     # next token_id

    return (x, dwell, elapsed, hour, dow, month, gap, age, recency, velocity, decay,
            tier, pop, p2p_score, not_bounce, cat, y, sample_weights)

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
        self.num_items = num_items
        self.num_categories = num_categories
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

        # Price tier embedding (Tier_1..Tier_5 + padding)
        self.tier_emb = nn.Embedding(6, tier_emb_dim, padding_idx=PAD_IDX)

        # Category embedding for dual path
        self.cat_emb = nn.Embedding(num_categories + 1, cat_emb_dim, padding_idx=PAD_IDX)

        # Day of week embedding (7 days, dim=4)
        if use_dow:
            self.dow_emb = nn.Embedding(7, 4)

        # Input dim calculation
        in_dim = item_emb_dim + tier_emb_dim
        if use_dwell: in_dim += 1
        if use_elapsed: in_dim += 1
        if use_tod: in_dim += 2   # sin + cos
        if use_dow: in_dim += 4   # embedding
        if use_month: in_dim += 2
        if use_gap: in_dim += 1
        if use_age: in_dim += 1
        if use_recency: in_dim += 1
        if use_velocity: in_dim += 1
        if use_decay: in_dim += 1
        if use_pop: in_dim += 1
        if use_p2p: in_dim += 1
        if use_not_bounce: in_dim += 1

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
        """
        x: (B, T) token indices
        tier: (B, T) price tier indices
        pop: (B, T) popularity score
        p2p_score: (B, T) transition strength
        not_bounce: (B, T) not-bounce flag
        cat: (B, T) category indices
        dwell: (B, T) normalized dwell times
        elapsed: (B, T) normalized elapsed times
        hour: (B, T, 2) sin, cos
        dow: (B, T) day of week index
        month: (B, T, 2) month sin/cos
        gap: (B, T) inter-session gap (normalized)
        age: (B, T) product age (normalized)
        recency: (B, T) product recency (normalized)
        velocity: (B, T) product velocity (normalized)
        decay: (B, T) time decay weights
        """
        e = self.item_emb(x)  # (B, T, E)
        t = self.tier_emb(tier) if tier is not None else torch.zeros_like(e[..., :0])

        feats_list = [e, t]
        if self.use_dwell and dwell is not None:
            feats_list.append(dwell.unsqueeze(-1))
        if self.use_elapsed and elapsed is not None:
            feats_list.append(elapsed.unsqueeze(-1))
        if self.use_tod and hour is not None:
            feats_list.append(hour)
        if self.use_dow and dow is not None:
            d = self.dow_emb(dow)
            feats_list.append(d)
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
        device='cpu',
        banned=None,
    ):
        """Predict top-k next items with temporal + product features."""
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

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
gru_model.to(device)
print(f"   Device: {device}")
print(f"   Parameters: {sum(p.numel() for p in gru_model.parameters()):,}")
print(f"   Dwell Time: {'Enabled' if USE_DWELL_TIME else 'Disabled'}")
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
model_path = 'model/model.pt'

for epoch in range(EPOCHS):
    gru_model.train()
    total_loss, n_batches = 0.0, 0

    for (x, dwell_b, elapsed_b, hour_b, dow_b, month_b, gap_b, age_b, recency_b, velocity_b,
         decay_b, tier_b, pop_b, p2p_b, not_bounce_b, cat_b, y, w) in train_loader:
        x = x.to(device)
        tier_b = tier_b.to(device)
        pop_b = pop_b.to(device)
        p2p_b = p2p_b.to(device)
        not_bounce_b = not_bounce_b.to(device)
        cat_b = cat_b.to(device)
        dwell_b = dwell_b.to(device)
        elapsed_b = elapsed_b.to(device)
        hour_b = hour_b.to(device)
        dow_b = dow_b.to(device)
        month_b = month_b.to(device)
        gap_b = gap_b.to(device)
        age_b = age_b.to(device)
        recency_b = recency_b.to(device)
        velocity_b = velocity_b.to(device)
        decay_b = decay_b.to(device)
        y = y.to(device)
        w = w.to(device)

        opt.zero_grad()
        logits = gru_model(
            x,
            tier_b,
            pop_b,
            p2p_b,
            not_bounce_b,
            cat_b,
            dwell_b,
            elapsed_b,
            hour_b,
            dow_b,
            month_b,
            gap_b,
            age_b,
            recency_b,
            velocity_b,
            decay_b,
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
# 7. TEST ENCODING WITH TEMPORAL FEATURES
# ==============================================================================
print("\n[7/8] Preparing test sequences with temporal features...")

# Encode test sequences with temporal features
test_actions['product_count'] = test_actions['user_actions'].apply(count_products_temporal)

test_encoded = {}
for _, row in test_actions.iterrows():
    vid = str(row['session_id'])
    if row['product_count'] >= 3:
        encoded = encode_sequence_with_temporal(
            row['user_actions'],
            pid2idx,
            p2p=p2p,
            p2p_totals=p2p_totals
        )
        if len(encoded) >= 2:
            test_encoded[vid] = encoded

total_test = len(test_actions)
print(f"   Total test sessions: {total_test:,}")
print(f"   Encoded (>=3 products): {len(test_encoded):,} ({len(test_encoded)/max(total_test,1)*100:.1f}%)")

# ==============================================================================
# 8. PREDICTION
# ==============================================================================
print("\n[8/8] Predicting...")

# Build test context from merged data
last_slug_by_visit = get_last_slug_from_merged(test_merged)
last_cat_by_visit = get_last_cat_from_merged(test_merged)
test_context = {}
for vid, seq in test_sequences_by_visit.items():
    test_context[vid] = {
        'last_slug': last_slug_by_visit.get(vid),
        'last_cat': last_cat_by_visit.get(vid),
        'product_seq': [int(pid) for pid in seq]
    }
for vid, slug in last_slug_by_visit.items():
    vid = str(vid)
    if vid not in test_context:
        test_context[vid] = {
            'last_slug': slug,
            'last_cat': last_cat_by_visit.get(vid),
            'product_seq': []
        }

targets = pd.read_parquet(f'{DATA_DIR}/metrika_visits_test.parquet', columns=['visit_id'])
print(f"   Total target visits: {len(targets):,}")
print(f"   Test context available: {len(test_context):,}")

preds = []
stats = {'coocc': 0, 'gru': 0, 'search': 0, 'global': 0}

for vid in tqdm(targets['visit_id']):
    recs = []
    def add_candidates(candidates):
        for cand in candidates:
            if cand not in recs:
                recs.append(cand)
            if len(recs) >= 6:
                break

    if vid in test_context:
        action = test_context[vid]
        slug = action['last_slug']
        product_seq = action['product_seq']

        # For 1-2 products: use scored_recommend (blends coocc + p2p + trigrams + cat2p + orders)
        if len(product_seq) in [1, 2] and len(recs) < 6:
            last_cat = action.get('last_cat')
            add_candidates(scored_recommend(
                product_seq, coocc, p2p, trigrams_dict, cat2p, order_cooccur,
                last_cat=last_cat, top_k=6
            ))
            if recs:
                stats['coocc'] += 1
        
        # For 3+ products: use Dual GRU
        if len(product_seq) >= 3 and len(recs) < 6:
            if vid in test_encoded:
                encoded_seq = test_encoded[vid]

                # Unpack temporal features
                unpacked = list(zip(*encoded_seq))
                session_tokens = list(unpacked[0])
                session_dwell = list(unpacked[1])
                session_elapsed = list(unpacked[2])
                session_hour = list(zip(unpacked[3], unpacked[4]))  # sin, cos
                session_dow = list(unpacked[5])
                session_month = list(zip(unpacked[6], unpacked[7]))  # month sin, cos
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
                    banned=set(session_tokens)
                )

                gru_pids = [str(idx2pid.get(idx, idx)) for idx in gru_preds if idx > 0]
                add_candidates(gru_pids)
                if recs:
                    stats['gru'] += 1

        # If GRU did not fill all slots, use co-occurrence fallback
        if len(product_seq) >= 3 and len(recs) < 6:
            before = len(recs)
            coocc_recs = coocc_recommend(
                [str(x) for x in product_seq],
                coocc,
                top_k=6,
                lookback=5
            )
            add_candidates(coocc_recs)
            if len(recs) > before:
                stats['coocc'] += 1
        
        if len(recs) < 6 and pd.notna(slug):
            # Primary: behavioral cat2p (empirical transitions)
            if slug in cat2p and cat2p[slug]:
                cat_recs = [str(k) for k, v in cat2p[slug].most_common(10)]
                add_candidates(cat_recs)
            # Fallback: text search
            if len(recs) < 6:
                query = SLUG_TO_CAT_MAP.get(slug, slug)
                search_results = search_products(query, n=10)
                add_candidates([str(s) for s in search_results])
            if not product_seq and recs:
                stats['search'] += 1

    if not recs:
        stats['global'] += 1
    if len(recs) < 6:
        add_candidates([x for x in global_top])

    preds.append({'visit_id': vid, 'product_ids': " ".join(recs[:6])})

sub = pd.DataFrame(preds)

# ---- Submission Preview (sanity check before saving) ----
total = len(sub)
print(f"\n{'='*60}")
print(f"  SUBMISSION PREVIEW")
print(f"{'='*60}")
print(f"  Shape: {sub.shape}")
print(f"  Columns: {list(sub.columns)}")
print(f"\n  First 10 rows:")
print(sub.head(10).to_string(index=False))
print(f"\n  Last 5 rows:")
print(sub.tail(5).to_string(index=False))

# Check for issues
n_recs = sub['product_ids'].apply(lambda x: len(str(x).split()))
n_empty = (sub['product_ids'].isna() | (sub['product_ids'] == '')).sum()
n_short = (n_recs < 6).sum()
n_dupes = sub['visit_id'].duplicated().sum()
print(f"\n  Recs per visit -- Min: {n_recs.min()} | Max: {n_recs.max()} | Mean: {n_recs.mean():.1f}")
print(f"  Empty predictions: {n_empty}")
print(f"  Visits with < 6 recs: {n_short}")
print(f"  Duplicate visit_ids: {n_dupes}")
if n_empty > 0 or n_short > 0 or n_dupes > 0:
    print(f"  ⚠️  Issues detected! Review before submitting.")
else:
    print(f"  ✅ All checks passed!")
print(f"{'='*60}")

# ---- Save ----
sub.to_csv(SUBMISSION_FILE, index=False)

print(f"\n{'='*60}")
print(f"  PREDICTION SUMMARY")
print(f"{'='*60}")
print(f"  Total predictions:   {total:,}")
print(f"  GRU (3+ products):   {stats['gru']:,} ({stats['gru']/total*100:.1f}%)")
print(f"  Coocc (1-2/fallback): {stats['coocc']:,} ({stats['coocc']/total*100:.1f}%)")
print(f"  Search/cat2p:        {stats['search']:,} ({stats['search']/total*100:.1f}%)")
print(f"  Global fallback:     {stats['global']:,} ({stats['global']/total*100:.1f}%)")
print(f"{'='*60}")
print(f"  Saved to: {SUBMISSION_FILE}")
