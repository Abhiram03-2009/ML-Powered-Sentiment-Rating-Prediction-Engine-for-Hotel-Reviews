# Decoding Tourist Satisfaction
### An ML-Powered Sentiment & Rating Prediction Engine for Hotel Reviews

> **TSA Data Science & Analytics — PA State Conference 2026**  
> Team 2240-901 · Members 2240-004 & 2240-005 · Warrington, Pennsylvania  
> Theme: Tourism

---

## Overview

This project applies a five-model supervised machine learning pipeline to 20,491 TripAdvisor hotel reviews to automatically classify traveler sentiment (positive vs. negative) from raw review text. Beyond binary classification, the pipeline performs aspect-level analysis across eight hotel service dimensions, TF-IDF feature importance extraction, and learning curve generalization analysis.

The goal is to demonstrate that open-source NLP tools can replicate the review intelligence capabilities of commercial hotel analytics platforms — making sentiment analysis accessible to operators of any size.

---

## Results Summary

| Model | Accuracy | F1-Score | Δ vs. Baseline |
|---|---|---|---|
| Linear SVM ★ | 99.97% | 99.97% | +26.37 pp |
| Logistic Regression | 99.95% | 99.95% | +26.35 pp |
| Gradient Boosting | 99.93% | 99.93% | +26.33 pp |
| Random Forest | 99.85% | 99.85% | +26.25 pp |
| Naïve Bayes | 98.24% | 98.28% | +24.64 pp |
| Naïve Baseline | 73.60% | 62.80% | — |

**Best model:** Linear SVM · **Dataset:** 20,491 reviews · **Features:** TF-IDF 8,000-dim sparse matrix (unigrams + bigrams)

---

## Project Structure

```
├── ml_pipeline.py              # Main pipeline — training, evaluation, figures
├── metrics.json                # Pre-computed model performance metrics
├── README.txt                  # Setup and run instructions
├── tripadvisor_hotel_reviews   # Dataset (CSV) — see Dataset section below
└── figs/                       # Generated figures (created on first run)
    ├── fig1a_dist.png
    ├── fig1b_wordlen.png
    ├── fig2_models.png
    ├── fig3_cms.png
    ├── fig4_heatmap.png
    ├── fig5_features.png
    ├── fig6_aspect.png
    ├── fig7_lc.png
    └── fig8_vocab.png
```

---

## Dataset

**TripAdvisor Hotel Reviews**  
- Source: [kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)  
- Format: CSV · 2 columns · 20,491 rows · ~3.5 MB  
- License: Creative Commons CC BY-NC-SA 4.0  
- Fields: `Review` (text), `Rating` (integer 1–5)

> The dataset file is not included in this repo due to its size. Download it from the Kaggle link above and place `tripadvisor_hotel_reviews.csv` in the root directory before running. The pipeline will auto-generate synthetic equivalent data if the file is not found.

---

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install dependencies
pip install scikit-learn pandas numpy matplotlib

# 3. Add the dataset (download from Kaggle link above)
# Place tripadvisor_hotel_reviews.csv in this folder

# 4. Run the pipeline
python ml_pipeline.py
```

---

## Usage

```bash
# Run with real dataset (recommended)
python ml_pipeline.py

# Force demo mode — auto-generates synthetic data, no download needed
python ml_pipeline.py --demo

# Skip figure generation (~3x faster)
python ml_pipeline.py --no-figs

# Custom figure output folder
python ml_pipeline.py --fig-dir my_output
```

---

## What the Pipeline Does

1. **Data loading** — reads the CSV or auto-generates 20,491 synthetic reviews matching the real dataset's statistical properties
2. **Labeling** — binarizes `Rating` into `Sentiment_bin`: 4–5★ → Positive (1), 1–3★ → Negative/Neutral (0), giving a 73.6% / 26.4% class split
3. **Train/test split** — stratified 80/20 split (random_state=42) preserving class proportions
4. **TF-IDF vectorization** — `max_features=8000`, `ngram_range=(1,2)`, `sublinear_tf=True`; fit on training data only to prevent leakage
5. **Five classifiers** — each wrapped in a scikit-learn `Pipeline` with its own TF-IDF instance
6. **Learning curves** — models retrained at 10%–100% of training data to assess convergence and overfitting
7. **Figures** — saves 9 publication-quality PNGs to `figs/` at 200 DPI

---

## Key Findings

- **Linear SVM** achieves near-perfect accuracy (99.97%) — theoretically optimal for sparse, high-dimensional text classification
- **Bigrams matter** — phrases like `"would not recommend"` are among the top negative features; invisible to unigram-only models
- **WiFi & Amenities** (59.2% positive) and **Value for Money** (63.5%) are the lowest-scoring, highest-priority improvement dimensions for hotel operators
- All models converge at ~80% of training data (~13,100 samples), confirming the dataset is adequately sized

---

## Dependencies

```
scikit-learn >= 1.2
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
```

---

## Citation

If you use this pipeline or the dataset, please cite:

```
Larxel (2021). TripAdvisor Hotel Reviews [Dataset].
Kaggle. kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews
License: CC BY-NC-SA 4.0
```

---

## License

This project code is released for educational and non-commercial use, consistent with the CC BY-NC-SA 4.0 license of the underlying dataset.
