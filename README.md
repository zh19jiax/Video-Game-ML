# Game Market Price Prediction System

An end-to-end ML pipeline that recommends pre-launch price points for video games from product attributes (genre, platform, length, quality ratings, etc.). Trained and evaluated on a 47,774-row catalog of game listings.

## Headline Results

| Model | R² | RMSE | Train time |
|---|---|---|---|
| Baseline (mean predictor) | 0.000 | 11.50 | — |
| Decision Tree (tuned) | 0.958 | 2.35 | ~1s |
| Random Forest | 0.960 | 2.31 | 3.3s |
| Neural Network (MLP-16) | 0.962 | 2.26 | 7.0s |
| **HistGradientBoosting (chosen)** | **0.962** | **2.24** | **1.1s** |
| GradientBoosting (sklearn) | 0.962 | 2.24 | 16.8s |

Gradient Boosting was selected for its best R² / runtime trade-off: same accuracy as the MLP and the slower sklearn GBM, but **~6× faster than the MLP** and **~3× faster than Random Forest**.

After correlation-based feature reduction (dropping the bottom 32% of features by |corr| with `Price`), R² is unchanged (0.962 → 0.962), confirming those features carried no signal.

## What This Project Does

1. **Ingests** the [Kaggle Video Game Reviews & Ratings dataset](https://www.kaggle.com/datasets/jahnavipaliwal/video-game-reviews-and-ratings) (47,774 rows × 18 columns).
2. **Cleans** the data through a reusable feature pipeline:
   - Drops identifier / high-cardinality columns (`Game Title`, `Developer`, `Publisher`, `User Review Text`, `Release Year`).
   - Imputes missing values (median for numeric, mode for categorical) — generalizes the pipeline to noisier real data even though this dataset is complete.
   - Removes outliers via Tukey IQR (`k = 3`).
   - Ordinal-encodes quality columns (`Graphics`/`Soundtrack`/`Story`) with a hand-mapped scale to preserve order; one-hot encodes nominal categoricals.
3. **Trains and compares** Baseline, Decision Tree, Random Forest, MLP, and two Gradient Boosting variants on the same 70/30 split.
4. **Prunes features** by absolute Pearson correlation with `Price` and verifies that dropping ~30% of the feature set does not degrade R².

## Repository Layout

```
.
├── data/
│   └── video_game_reviews.csv          # raw input (47,774 rows)
├── src/
│   ├── preprocessing.py                # build_features(): impute → outlier → encode
│   └── evaluate.py                     # fit_score(): timed training + R²/RMSE
├── notebooks/
│   ├── 01_eda.ipynb                    # correlation matrix, rating distribution
│   ├── 02_preprocessing.ipynb          # walks through the feature pipeline
│   ├── 03_knn_baseline.ipynb           # KNN sweep over k (preliminary)
│   ├── 04_decision_tree_random_forest.ipynb
│   ├── 05_neural_network.ipynb         # MLP regressor
│   ├── 06_gradient_boosting.ipynb      # HistGBR + sklearn GBM
│   ├── 07_feature_reduction.ipynb      # correlation pruning, retrain GBM
│   └── 08_model_comparison.ipynb       # head-to-head benchmark
├── reports/
│   ├── final_report.md                 # course final report (methodology, rationale)
│   ├── model_comparison.csv            # generated benchmark table
│   └── feature_reduction.csv           # generated reduction comparison
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Run the full model comparison
jupyter notebook notebooks/08_model_comparison.ipynb

# Or train programmatically
python -c "
from src.preprocessing import build_features
from src.evaluate import fit_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

X, y, _ = build_features('data/video_game_reviews.csv')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=67)
print(fit_score(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=8, random_state=0), Xtr, Xte, ytr, yte, 'GBM').as_row())
"
```

## Why Gradient Boosting Wins Here

The tree-based ensembles, the MLP, and HistGBR all land in a narrow R² band (0.958–0.962). Two practical reasons HistGBR is the deployment choice:

- **Runtime.** 1.1s end-to-end training vs. 7s for the MLP and 17s for sklearn's classic GBM, on the same machine. Matters for iterative tuning and retraining as the catalog grows.
- **Native handling of mixed-type features.** HistGBR bins numeric features and treats encoded categoricals without scaling, so the feature pipeline stays simple — no `StandardScaler` step needed.

The MLP's performance (R² = 0.962) is competitive but not interpretable, harder to debug, and slower to retrain.

## Honest Caveats (from the final report)

- **The dataset is partially synthetic.** Numeric columns like `Price`, `User Rating`, and `Game Length` look uniformly distributed, which is unusual for real markets. The model would need validation against organic data (Steam, PlayStation Store, Metacritic) before production use. See `reports/final_report.md` §6d.
- **`User Rating` dominates** the feature importance (|corr| with Price = 0.76). The R² ≈ 0.96 across multiple model families is largely driven by this one feature. The remaining lift from genre/platform/length is real but modest. See `reports/final_report.md` §6b.

## Course Context

Originally built as the final project for **BU BA305 (Data Mining for Business Analytics)** with Humphrey Wang, Jiarui Yu, and Xinni Cai. This version extends the original submission with:

- A reusable, testable feature pipeline (`src/preprocessing.py`).
- Gradient Boosting models (not in the original submission).
- A correlation-based feature reduction experiment.
- A reorganized repository structure suitable as a portfolio piece.

See [`reports/final_report.md`](reports/final_report.md) for the full course writeup including dataset selection rationale, preliminary model comparisons, and discussion of real-world applications.
