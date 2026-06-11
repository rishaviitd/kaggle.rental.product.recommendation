# Kaggle Rental Product Recommendation

🏆 **Best Score on Kaggle (Recall@6: 0.417)**
* **Notebook:** [Rental Product Recommendation GRU](https://www.kaggle.com/code/atomstack001/rental-product-recommendation-gru)
* **Competition:** [Rental Product Recommendation System](https://www.kaggle.com/competitions/rental-product-recommendation-system)

## 🚀 Quick Start (Reproduce in 3 Steps)

This project uses [`uv`](https://docs.astral.sh/uv/) to guarantee 100% dependency reproducibility.

```bash
# 1. Clone the repository
git clone https://github.com/rishaviitd/kaggle.rental.product.recommendation.git
cd kaggle.rental.product.recommendation

# 2. Instantly recreate the exact locked environment
uv sync

# 3. Train model and save inference artifacts
uv run train.py

# 4. Generate output/predictions.csv from saved artifacts
uv run inference.py
```

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
