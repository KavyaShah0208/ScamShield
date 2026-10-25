"""
Explainability engine for ScamShield.
Calculates exact linear local feature contributions (w_i * x_i) from Logistic Regression
and identifies non-overlapping character spans in the input message.
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np

def compute_local_explanations(
    text: str,
    vectorizer,
    classifier,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Computes exact local feature attribution for a single message.
    
    Formula:
      logit = intercept + sum(w_i * x_i)
      token_contribution = w_i * x_i
    
    Returns:
    - predicted_class: 0 (legitimate) or 1 (scam)
    - probabilities: [prob_ham, prob_spam]
    - token_attributions: list of (token, weight, contribution)
    - highlighted_spans: character offset spans for UI rendering
    """
    # 1. Transform text
    try:
        from src.preprocessing import preprocess_text
    except ImportError:
        from preprocessing import preprocess_text
    cleaned = preprocess_text(text)
    x_vec = vectorizer.transform([cleaned])
    
    # 2. Classifier inference
    prob_spam = float(classifier.predict_proba(x_vec)[0, 1])
    predicted_class = int(prob_spam >= 0.5)

    # 3. Exact linear feature attribution
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    
    # Non-zero entries in the sparse row
    x_coo = x_vec.tocoo()
    contributions = []
    
    for idx, tfidf_val in zip(x_coo.col, x_coo.data):
        feat = feature_names[idx]
        weight = coefs[idx]
        local_contrib = weight * tfidf_val
        contributions.append({
            "feature": feat,
            "weight": round(float(weight), 4),
            "tfidf": round(float(tfidf_val), 4),
            "contribution": round(float(local_contrib), 4)
        })

    # Sort descending by positive contribution toward scam
    contributions.sort(key=lambda item: item["contribution"], reverse=True)
    
    # Select top features that push towards scam
    scam_features = [c for c in contributions if c["contribution"] > 0][:top_k]
    # Select features that push towards legitimate (ham)
    ham_features = [c for c in contributions if c["contribution"] < 0][:top_k]

    # 4. Map top scam features back to character spans in original text
    spans = []
    for item in scam_features:
        feat = item["feature"]
        # Handle n-grams by escaping
        pattern = re.compile(re.escape(feat), re.IGNORECASE)
        for match in pattern.finditer(text):
            spans.append({
                "start": match.start(),
                "end": match.end(),
                "text": match.group(0),
                "feature": feat,
                "contribution": item["contribution"],
                "source": "ML Model"
            })

    # Deduplicate and resolve overlapping spans (keep the one with higher contribution)
    resolved_spans = _resolve_overlapping_spans(spans)

    return {
        "predicted_class": predicted_class,
        "is_scam": bool(predicted_class == 1),
        "risk_score": round(prob_spam, 4),
        "scam_contributions": scam_features,
        "ham_contributions": ham_features,
        "resolved_spans": resolved_spans,
        "intercept": round(float(classifier.intercept_[0]), 4)
    }

def _resolve_overlapping_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sorts and resolves any overlapping character intervals."""
    if not spans:
        return []

    # Sort by start position asc, then contribution desc
    spans = sorted(spans, key=lambda s: (s["start"], -s["contribution"]))
    resolved = []
    
    current = spans[0]
    for nxt in spans[1:]:
        # If overlapping with current span
        if nxt["start"] < current["end"]:
            # If current span has higher contribution, keep current; else replace
            if nxt["contribution"] > current["contribution"]:
                current = nxt
        else:
            resolved.append(current)
            current = nxt
    resolved.append(current)
    return resolved

def build_highlighted_html(raw_text: str, ml_spans: List[Dict[str, Any]], rule_findings: List[Dict[str, Any]]) -> str:
    """
    Renders safe, accessible inline HTML highlights within the raw text.
    Combines both ML model attribution spans and Rule-based indicator spans.
    """
    import html

    # Merge all spans
    all_spans = []
    for s in ml_spans:
        all_spans.append({
            "start": s["start"],
            "end": s["end"],
            "label": f"ML Model (+{s['contribution']})",
            "type": "ml"
        })

    for rule in rule_findings:
        for m in rule["matches"]:
            all_spans.append({
                "start": m["start"],
                "end": m["end"],
                "label": f"Indicator: {rule['category']}",
                "type": "rule"
            })

    if not all_spans:
        return f'<div class="scamshield-highlight-box">{html.escape(raw_text)}</div>'

    # Deduplicate overlapping spans
    all_spans.sort(key=lambda s: s["start"])
    clean_spans = []
    last_end = 0
    for s in all_spans:
        if s["start"] >= last_end:
            clean_spans.append(s)
            last_end = s["end"]

    # Construct HTML string
    out = []
    curr_idx = 0
    for s in clean_spans:
        # Preceding unhighlighted text
        if s["start"] > curr_idx:
            out.append(html.escape(raw_text[curr_idx:s["start"]]))
        
        # Highlighted chunk
        chunk = html.escape(raw_text[s["start"]:s["end"]])
        cls_name = "hl-ml" if s["type"] == "ml" else "hl-rule"
        badge_cls = "badge-ml" if s["type"] == "ml" else "badge-rule"
        out.append(
            f'<mark class="{cls_name}" title="{s["label"]}">{chunk}'
            f'<span class="{badge_cls}"></span></mark>'
        )
        curr_idx = s["end"]

    # Trailing unhighlighted text
    if curr_idx < len(raw_text):
        out.append(html.escape(raw_text[curr_idx:]))

    return f'<div class="scamshield-highlight-box">{"".join(out)}</div>'
