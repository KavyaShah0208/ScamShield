# ScamShield — Explainable Scam & Phishing Message Analyzer

A local, privacy-preserving security utility that analyzes SMS and email messages for phishing and scam patterns using a trained Machine Learning model and passive security heuristics.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Virtual environment (`.venv`)

### 2. Activate Environment & Run Application
```bash
# In the project directory:
source .venv/bin/activate

# Launch ScamShield web utility:
streamlit run app.py --server.port=8502
```
Access the application in your browser at: **`http://localhost:8502`**

---

## Key Features

- **Local Machine Learning Classification:** Trained Logistic Regression model with TF-IDF unigram and bigram vectorization.
- **Explainable Predictions:** Exact local log-odds token attribution highlighting suspicious phrases directly in the original message.
- **Rule-Based Security Indicators:** Independent detection of artificial urgency, credential harvesting, OTP prompts, and financial lures.
- **Zero-Network URL Inspection:** Analyzes links for raw IP hostnames, link shorteners, and protocol anomalies without making any outbound network or DNS requests.
- **Privacy First:** 100% offline execution on localhost; messages are never sent to external servers or cloud APIs.

---

## Project Structure

```
ml project/
├── app.py                   # Streamlit web application
├── style.css                # Minimalist warm-neutral design system
├── requirements.txt         # Project dependencies
├── models/
│   ├── classifier.pkl       # Trained Logistic Regression model
│   ├── baseline_nb.pkl      # Baseline Multinomial Naive Bayes model
│   ├── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
│   └── evaluation_meta.json # Evaluation metrics & confusion matrix
├── src/
│   ├── explainability.py    # Feature attribution & HTML span highlighter
│   ├── indicators.py        # Passive URL analysis & rule indicators
│   ├── preprocessing.py     # Text cleaning & normalization
│   ├── prepare_data.py      # Dataset loading & stratified split
│   └── train.py             # Reproducible training pipeline
└── data/
    ├── raw/                 # UCI SMS Spam Collection
    └── processed/           # Stratified train (80%) and test (20%) splits
```

---

## Retraining the Model

To re-run data preparation and model training from scratch:
```bash
source .venv/bin/activate
python src/prepare_data.py
python src/train.py
```
