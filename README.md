# Kaggle Rental Product Recommendation

🏆 **Best Score on Kaggle (Recall@6: 0.414)**
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

The system operates as a **Multi-Tiered Recommender**. The core of the system is a Temporal Dual-Path GRU (Gated Recurrent Unit) neural network that predicts the next item in a sequence based on recent user clicks and the time spent on those items. 

Because neural networks struggle with very short sessions (e.g., 1 or 2 clicks), the system seamlessly degrades into statistical co-occurrence (P2P), behavioral mappings, text-search indexing, and global popularity to guarantee 6 high-quality recommendations for every single visit.

---

## 1. Data Preprocessing & Mapping

Before any machine learning happens, the data from the old site and the new site must be unified.

* **Product Unification**
  * Consolidates product IDs and URL slugs from legacy datasets and the modern platform into a single ground-truth mapping.
  * Ensures that historical interactions on discontinued URLs accurately map to current active product IDs.

* **Category Learning**
  * Dynamically learns missing category mappings by observing what products users click immediately after visiting a category page.
  * Fixes sparse category data by letting actual user navigation behavior define what category a product belongs to.

## 2. Feature Engineering

The model doesn't just look at *what* the user clicked, it looks at *how* they clicked it.

* **Temporal Features**
  * Calculates dwell time (how long a user looked at an item), inter-session gaps, and applies mathematical time-decay to older actions.
  * Encodes cyclical time elements (Time of Day, Day of Week) so the model can learn if certain items are rented more on weekends or evenings.

* **Behavioral & Item Features**
  * Precomputes global popularity scores and specific item-to-item (P2P) transition strengths over the last 6 months.
  * Embeds static item metadata like price tiers and categories directly into the feature space for the model to utilize.

## 3. Core Model: Temporal Dual-Path GRU

The primary prediction engine is a custom PyTorch sequence model.

* **Item Path**
  * Feeds the sequence of clicked product IDs (along with temporal features like dwell time and decay) into a dedicated GRU layer.
  * Learns complex, deep relationships between specific items based on chronological user journeys.

* **Category Path**
  * Simultaneously feeds the sequence of product categories into a parallel, secondary GRU layer.
  * Allows the model to recognize high-level intent (e.g., "this user is looking at strollers") even if it hasn't seen the specific item IDs before.

## 4. Multi-Tiered Inference Engine

When predicting the next 6 items for a test visit, the system evaluates the session length and chooses the best available strategy.

* **Tier 1: GRU Predictions (Rich Sessions)**
  * Used for sessions with 3 or more product interactions, leveraging the trained neural network to generate highly contextual recommendations.
  * Provides the most accurate, personalized results for deeply engaged users.

* **Tier 2: Co-occurrence & Transitions (Short Sessions)**
  * Used for sessions with only 1 or 2 interactions by relying on statistical Item-to-Item (P2P) transition matrices built from the training data.
  * Recommends the items that most frequently follow the specific product the user just looked at.

* **Tier 3: Search & Behavioral Fallback (Cold Start)**
  * Uses a custom text-search index to find products matching the URL slugs, or recommends popular items from the last viewed category (cat2p).
  * Prevents empty recommendations when the user interacts with brand new items that lack historical data.

* **Tier 4: Global Popularity (Absolute Fallback)**
  * Fills any remaining recommendation slots with the most universally popular products rented over the last 6 months.
  * Guarantees that the system always outputs exactly 6 valid predictions, maximizing baseline hit rates.
