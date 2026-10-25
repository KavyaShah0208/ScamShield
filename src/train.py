"""
Training pipeline for ScamShield (College ML Experiment 14).
Trains both a Multinomial Naive Bayes baseline and an explainable Logistic Regression classifier.
Evaluates strictly on held-out test data without leakage and exports artifacts.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from preprocessing import preprocess_text

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

def train_and_evaluate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed train or test split missing. Run prepare_data.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded Train set: {len(train_df)} rows")
    print(f"Loaded Test set:  {len(test_df)} rows")

    # Apply preprocessing
    train_clean = [preprocess_text(msg) for msg in train_df['message']]
    test_clean = [preprocess_text(msg) for msg in test_df['message']]

    y_train = train_df['target'].values
    y_test = test_df['target'].values

    # Fit TF-IDF Vectorizer ONLY on training set (avoids data leakage)
    print("\nFitting TF-IDF Vectorizer (unigrams + bigrams, min_df=2, sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words='english'
    )
    X_train = vectorizer.fit_transform(train_clean)
    X_test = vectorizer.transform(test_clean)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")

    # --- 1. Baseline: Multinomial Naive Bayes ---
    print("\n--- Training Model 1: Multinomial Naive Bayes (Baseline) ---")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)

    nb_pred = nb_model.predict(X_test)
    nb_prob = nb_model.predict_proba(X_test)[:, 1]
    nb_prec, nb_rec, nb_f1, _ = precision_recall_fscore_support(y_test, nb_pred, average='binary')
    nb_cm = confusion_matrix(y_test, nb_pred).tolist()
    nb_auc = roc_auc_score(y_test, nb_prob)

    print(f"Naive Bayes -> Precision: {nb_prec:.4f}, Recall: {nb_rec:.4f}, F1: {nb_f1:.4f}, ROC-AUC: {nb_auc:.4f}")

    # --- 2. Primary: Logistic Regression (Explainable Champion) ---
    print("\n--- Training Model 2: Logistic Regression (Champion) ---")
    lr_model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average='binary')
    lr_cm = confusion_matrix(y_test, lr_pred).tolist()
    lr_auc = roc_auc_score(y_test, lr_prob)

    print(f"Logistic Regression -> Precision: {lr_prec:.4f}, Recall: {lr_rec:.4f}, F1: {lr_f1:.4f}, ROC-AUC: {lr_auc:.4f}")
    print("\nDetailed Classification Report (Logistic Regression):")
    print(classification_report(y_test, lr_pred, target_names=['Legitimate (Ham)', 'Scam/Phishing (Spam)']))

    # Top predictive features for interpretability
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = lr_model.coef_[0]
    top_spam_indices = np.argsort(coefs)[-20:][::-1]
    top_ham_indices = np.argsort(coefs)[:20]

    top_spam_features = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_spam_indices]
    top_ham_features = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_ham_indices]

    # Save artifacts
    print("\nSaving model artifacts to models/...")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(lr_model, MODELS_DIR / "classifier.pkl")
    joblib.dump(nb_model, MODELS_DIR / "baseline_nb.pkl")

    # Save comprehensive evaluation metadata for UI and practical report
    eval_meta = {
        "dataset": "UCI SMS Spam Collection",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "baseline_naive_bayes": {
            "precision": round(float(nb_prec), 4),
            "recall": round(float(nb_rec), 4),
            "f1_score": round(float(nb_f1), 4),
            "roc_auc": round(float(nb_auc), 4),
            "confusion_matrix": {
                "tn": nb_cm[0][0], "fp": nb_cm[0][1],
                "fn": nb_cm[1][0], "tp": nb_cm[1][1]
            }
        },
        "logistic_regression": {
            "precision": round(float(lr_prec), 4),
            "recall": round(float(lr_rec), 4),
            "f1_score": round(float(lr_f1), 4),
            "roc_auc": round(float(lr_auc), 4),
            "confusion_matrix": {
                "tn": lr_cm[0][0], "fp": lr_cm[0][1],
                "fn": lr_cm[1][0], "tp": lr_cm[1][1]
            }
        },
        "top_spam_indicators": top_spam_features,
        "top_ham_indicators": top_ham_features
    }

    with open(MODELS_DIR / "evaluation_meta.json", "w", encoding="utf-8") as f:
        json.dump(eval_meta, f, indent=2)

    print("Model artifacts and evaluation_meta.json exported successfully.")

if __name__ == "__main__":
    train_and_evaluate()
