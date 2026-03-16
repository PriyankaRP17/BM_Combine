# Mental Health Prediction Using Twitter Data

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-FF6600)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![VADER](https://img.shields.io/badge/NLP-VADER_Sentiment-4B8BBE)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> A two-stage NLP classification system that detects mental health conditions from text — first predicts if a post indicates a mental health concern, then classifies it into one of 7 categories with 84% binary accuracy and 0.93 average AUC.

---

## What It Does

Takes a user-written statement or tweet as input and runs it through a two-stage pipeline:

1. **Stage 1 — Binary Classification:** Is this post mentally healthy or not? (84% accuracy, AUC 0.93)
2. **Stage 2 — Multiclass Classification:** If a concern is detected, which category? (7 classes, AUC 0.94 average)

---

## Mental Health Categories

`Anxiety` · `Depression` · `Bipolar` · `Stress` · `Suicidal` · `Personality Disorder` · `Normal`

---

## Model Performance

| Model | Task | Accuracy | AUC |
|---|---|---|---|
| XGBoost (Binary) | Mental concern vs Normal | **84%** | **0.93** |
| XGBoost (Multiclass) | 7-class classification | **73%** | **0.94 avg** |

### Per-Class AUC (Multiclass)

| Class | AUC |
|---|---|
| Normal | 0.9786 |
| Bipolar | 0.9638 |
| Anxiety | 0.9579 |
| Personality Disorder | 0.9454 |
| Stress | 0.9262 |
| Suicidal | 0.9093 |
| Depression | 0.8840 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML Model | XGBoost (binary + multiclass) |
| Text Vectorization | TF-IDF (`max_features=10000`) |
| Dimensionality Reduction | TruncatedSVD (301 components binary, 100 multiclass) |
| Sentiment Analysis | VADER (`vaderSentiment`) |
| Encoding | LabelEncoder (multiclass) |
| Evaluation | Scikit-learn — classification report, confusion matrix, ROC-AUC |
| Model Persistence | `joblib` |
| Visualization | Matplotlib, Seaborn |

---

## Pipeline

```
Raw text input
      ↓
TF-IDF Vectorization (10,000 features)
      ↓
TruncatedSVD — dimensionality reduction
      ↓
VADER Sentiment Score — appended as extra feature
      ↓
XGBoost Binary Classifier
      ↓
If mental concern detected → XGBoost Multiclass Classifier
      ↓
Predicted category + probabilities

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/PriyankaRS17/BM_Combine.git

```

### Requirements

```
pandas
numpy
scikit-learn
xgboost
vaderSentiment
matplotlib
seaborn
joblib
jupyter
```

---

## Sample Prediction

```python
Input:  "happy moments #memories filled with fun"

Stage 1 → Mental concern probability: 0.53 → Concern detected
Stage 2 → Predicted Class: Normal
         Probabilities: Normal: 98.5%, Depression: 0.9%, Suicidal: 0.4%
```

---

## Key Engineering Decisions

- **Two-stage pipeline** — binary classifier acts as a gate before multiclass; reduces false positives in the sensitive 7-class model by only running it when a concern is likely
- **TF-IDF + SVD** — TF-IDF captures word importance across the corpus; SVD reduces 10,000 features to ~100–300 dense components, making XGBoost training fast and stable
- **VADER sentiment as augmented feature** — compound sentiment score appended to the reduced text vector; adds emotional tone signal that pure bag-of-words misses
- **XGBoost over traditional classifiers** — gradient boosting handles class imbalance better than logistic regression and outperforms SVM on high-dimensional sparse data after SVD reduction
- **Separate vectorizers per stage** — binary and multiclass models use independent TF-IDF and SVD pipelines, preventing data leakage between the two classification tasks

---

## What I'd Improve Next

- [ ] Address class imbalance (Stress and Personality Disorder have fewer samples) using SMOTE or class weights
- [ ] Replace TF-IDF with BERT embeddings for richer semantic features
- [ ] Build a Flask web interface for real-time text input and prediction
- [ ] Add explainability — SHAP values to show which words drove the prediction
- [ ] Deploy as a REST API for integration with other applications

---

## Author

**Priyanka R P** · [LinkedIn](https://www.linkedin.com/in/priyanka-rp) · [GitHub](https://github.com/PriyankaRS17) · priyankapremnath17@gmail.com
