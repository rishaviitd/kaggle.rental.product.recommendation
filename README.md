# Kaggle Rental Product Recommendation

🏆 **Best Score on Kaggle (Recall@6: 0.417)**
* **Notebook:** [Rental Product Recommendation GRU](https://www.kaggle.com/code/atomstack001/rental-product-recommendation-gru)
* **Competition:** [Rental Product Recommendation System](https://www.kaggle.com/competitions/rental-product-recommendation-system)

## Reproduce The Output

This project uses [`uv`](https://docs.astral.sh/uv/) to guarantee 100% dependency reproducibility.
The training pipeline is managed with DVC so generated artifacts can be reproduced from the same code, params, and data.

```bash
git clone https://github.com/rishaviitd/kaggle.rental.product.recommendation.git
cd kaggle.rental.product.recommendation

uv sync
```

### Data Requirement

Raw competition data is not committed to this repository.
Before running the pipeline, place the required parquet files under `data/`:

```text
data/metrika_hits.parquet
data/metrika_visits.parquet
data/metrika_hits_test.parquet
data/metrika_visits_test.parquet
data/products_all.parquet
data/old_site_products.parquet
data/new_site_products.parquet
data/old_site_new_site_products.parquet
data/new_site_orders.parquet
data/old_site_orders.parquet
```

### Rebuild Artifacts Locally

Use this when you have the raw `data/` files and want to regenerate all intermediate and final artifacts.

```bash
uv run dvc repro --force
uv run inference.py
```

This creates:

```text
artifacts/intermediate/
artifacts/final/
output/predictions.csv
```

### Restore Artifacts From A DVC Remote

Use this when you have access to a DVC remote. The remote must be configured first.
It can point to S3 or to a local artifact/cache directory, depending on your setup.

Example S3 remote:

```bash
uv run dvc remote add -d artifacts-s3 s3://your-bucket/path/to/dvc-cache
```

Example local remote:

```bash
uv run dvc remote add -d artifacts-local /path/to/local/dvc-cache
```

Then restore artifacts and generate predictions:

```bash
uv run dvc pull
uv run inference.py
```

The Git repository stores code, `params.yaml`, `dvc.yaml`, and `dvc.lock`.
DVC stores generated artifacts listed as pipeline outputs.
If you want to version your own raw data with DVC, configure your own remote and run `dvc add` for the files under `data/`.

This repository contains a hybrid recommendation system built to predict the next rental product a user will interact with based on their browsing session history.

The solution heavily leverages sequence modeling alongside robust fallback strategies to handle everything from rich, long-term user histories down to complete cold-starts.

## Overall Architecture

The system operates as a **Multi-Tiered Recommender**. The core of the system is a Dual-Path GRU (Gated Recurrent Unit) neural network that predicts the next item in a sequence based on recent product clicks, price tier, and category context.

Because neural networks struggle with very short sessions (e.g., 1 or 2 clicks), the system falls back to simpler statistical strategies based on how many products the user actually viewed. This routing guarantees 6 high-quality recommendations for every single visit, even for brand-new products with no history.

```mermaid
flowchart TD
    VISIT["User Visit<br/>(browsing session)"] --> Q{"How many products<br/>were viewed?"}

    Q -->|"3 or more"| GRU["Dual-Path GRU<br/>personalized predictions"]
    Q -->|"1 or 2"| STAT["Statistical Tiers<br/>pairwise transitions (what comes next)<br/>trigrams (3-item click patterns)<br/>co-occurrence (items viewed together)"]
    Q -->|"0 / unknown items"| COLD["Cold-Start Tiers<br/>inverted index search +<br/>category-to-product popularity"]

    GRU --> FILL{"Have 6<br/>recommendations?"}
    STAT --> FILL
    COLD --> FILL

    FILL -->|"No"| POP["Global Popularity<br/>most-rented products"]
    FILL -->|"Yes"| OUT["Final 6 Recommendations"]
    POP --> OUT

    classDef entry fill:#eef2ff,stroke:#6366f1,color:#1e293b;
    classDef decision fill:#fef9c3,stroke:#eab308,color:#1e293b;
    classDef model fill:#dcfce7,stroke:#22c55e,color:#1e293b;
    classDef fallback fill:#ffedd5,stroke:#f97316,color:#1e293b;
    classDef output fill:#cffafe,stroke:#06b6d4,color:#1e293b;

    class VISIT entry;
    class Q,FILL decision;
    class GRU model;
    class STAT,COLD,POP fallback;
    class OUT output;
```

The session length decides which tier handles the visit, and unfilled slots always cascade down to global popularity:

* **Tier 1: GRU Predictions (Rich Sessions)**
  * Used for sessions with 3 or more product interactions, leveraging the trained neural network for highly contextual, personalized recommendations.

* **Tier 2: Co-occurrence & Transitions (Short Sessions)**
  * Used for 1–2 interaction sessions, relying on pairwise transition tables (what product usually comes next), trigrams (common 3-item click patterns), and co-occurrence (items often viewed together).

* **Tier 3: Search & Behavioral Fallback (Cold Start)**
  * Queries an inverted index over product-URL keywords to match cold-start items, then falls back to category-to-product popularity from the last viewed category.

* **Tier 4: Global Popularity (Absolute Fallback)**
  * Fills any remaining slots with the most-rented products overall, guaranteeing exactly 6 valid predictions for every visit.

---

## Dual Path GRU Architecture

The primary prediction engine is a custom PyTorch sequence model built from compact inputs that proved most useful in ablations.

* **GRU Inputs**
  * Product token sequence from merged browsing sessions.
  * Price tier embedding for each clicked product.
  * Category sequence for the parallel category GRU path.

* **Training-Time Recency Weighting**
  * Applies exponential sample weighting on session age so more recent browsing sessions contribute more to the training loss.

```mermaid
flowchart TD
    subgraph Inputs
        X["Product Token Sequence<br/>(B, T)"]
        TIER["Price Tier Sequence<br/>(B, T)"]
        CAT["Category Sequence<br/>(B, T)"]
    end

    subgraph ItemPath["Item Path"]
        IE["Item Embedding<br/>(128-d)"]
        TE["Tier Embedding<br/>(4-d)"]
        CONCAT["Concat<br/>item + tier"]
        IPROJ["Linear -> ReLU -> Dropout<br/>(128-d)"]
        IGRU["Item GRU<br/>(hidden 128)"]
    end

    subgraph CatPath["Category Path"]
        CE["Category Embedding<br/>(8-d)"]
        CPROJ["Linear -> ReLU -> Dropout<br/>(96-d)"]
        CGRU["Category GRU<br/>(hidden 96)"]
    end

    FUSE["Concat item_h + cat_h<br/>(224-d)"]
    OUT["Output Linear<br/>(num_items + 1)"]
    SCORES["Next-Item Scores"]

    X --> IE
    TIER --> TE
    IE --> CONCAT
    TE --> CONCAT
    CONCAT --> IPROJ --> IGRU --> FUSE

    CAT --> CE --> CPROJ --> CGRU --> FUSE

    FUSE --> OUT --> SCORES

    classDef input fill:#eef2ff,stroke:#6366f1,color:#1e293b;
    classDef item fill:#dcfce7,stroke:#22c55e,color:#1e293b;
    classDef cat fill:#fae8ff,stroke:#d946ef,color:#1e293b;
    classDef head fill:#cffafe,stroke:#06b6d4,color:#1e293b;

    class X,TIER,CAT input;
    class IE,TE,CONCAT,IPROJ,IGRU item;
    class CE,CPROJ,CGRU cat;
    class FUSE,OUT,SCORES head;
```

* **Item Path**
  * Feeds the sequence of clicked product IDs and price tier embeddings into a dedicated GRU layer.
  * Learns relationships between specific items based on chronological user journeys.

* **Category Path**
  * Simultaneously feeds the sequence of product categories into a parallel, secondary GRU layer.
  * Allows the model to recognize high-level intent (e.g., "this user is looking at strollers") even if it hasn't seen the specific item IDs before.
