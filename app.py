"""
ScamShield — Explainable Scam and Phishing Message Analyzer
ChatGPT-inspired clean interface: simple prompt input, press Enter to analyze.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

from src.indicators import detect_security_indicators, extract_urls, analyze_url
from src.explainability import compute_local_explanations, build_highlighted_html

# Page Configuration
st.set_page_config(
    page_title="ScamShield",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load Custom CSS
css_path = Path("style.css")
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cache Models
@st.cache_resource
def load_models():
    models_dir = Path("models")
    vec_path = models_dir / "tfidf_vectorizer.pkl"
    clf_path = models_dir / "classifier.pkl"
    meta_path = models_dir / "evaluation_meta.json"

    if not vec_path.exists() or not clf_path.exists():
        return None, None, None

    vectorizer = joblib.load(vec_path)
    classifier = joblib.load(clf_path)
    
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return vectorizer, classifier, metadata

vectorizer, classifier, eval_meta = load_models()

# Manage Conversation Messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# Sidebar for Model Transparency & Settings
with st.sidebar:
    st.markdown("### ScamShield System")
    st.caption("Locally trained model on 5,160 verified messages. 100% offline analysis.")
    
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if eval_meta:
        st.markdown("---")
        st.markdown("#### Test Set Performance ($N=1,032$)")
        lr_meta = eval_meta.get("logistic_regression", {})
        st.write(f"• **Precision:** {lr_meta.get('precision', 0)*100:.1f}%")
        st.write(f"• **Recall:** {lr_meta.get('recall', 0)*100:.1f}%")
        st.write(f"• **F1-Score:** {lr_meta.get('f1_score', 0)*100:.1f}%")
        st.write(f"• **ROC-AUC:** {lr_meta.get('roc_auc', 0):.4f}")
        
        with st.expander("Confusion Matrix"):
            cm = lr_meta.get("confusion_matrix", {})
            st.dataframe(pd.DataFrame([
                {"True": "Legitimate", "Pred Ham": cm.get("tn"), "Pred Scam": cm.get("fp")},
                {"True": "Scam", "Pred Ham": cm.get("fn"), "Pred Scam": cm.get("tp")}
            ]), hide_index=True)

# Helper function to perform analysis
def analyze_message_content(text: str):
    urls = extract_urls(text)
    url_analyses = [analyze_url(u) for u in urls]
    has_suspicious_url = any(a["is_suspicious"] for a in url_analyses)
    rule_findings = detect_security_indicators(text)
    ml_result = compute_local_explanations(text, vectorizer, classifier)

    is_scam_ml = ml_result["is_scam"]
    risk_score = ml_result["risk_score"]

    if is_scam_ml or (risk_score >= 0.45 and (has_suspicious_url or rule_findings)):
        status_code = "high-risk"
        status_label = "HIGH RISK DETECTED"
        summary_text = "Strong indicators of a scam or phishing attempt. Flagged by both statistical model tokens and heuristic pattern detection."
    elif risk_score >= 0.25 or rule_findings or has_suspicious_url:
        status_code = "medium-risk"
        status_label = "SUSPICIOUS / ELEVATED CAUTION"
        summary_text = "Elevated risk indicators observed. Caution recommended before clicking links or replying."
    else:
        status_code = "low-risk"
        status_label = "LIKELY LEGITIMATE"
        summary_text = "No common scam, urgency, or credential phishing indicators detected."

    highlighted_html = build_highlighted_html(text, ml_result["resolved_spans"], rule_findings)

    return {
        "text": text,
        "status_code": status_code,
        "status_label": status_label,
        "risk_score": risk_score,
        "summary_text": summary_text,
        "highlighted_html": highlighted_html,
        "ml_result": ml_result,
        "rule_findings": rule_findings,
        "url_analyses": url_analyses,
        "urls": urls
    }

# Main View: Hero when empty, or Chat History when active
if not st.session_state.chat_history:
    st.markdown("""
    <div class="hero-container">
      <h1 class="hero-title">What message would you like to inspect?</h1>
      <p class="hero-subtitle">
        Paste any SMS or email. ScamShield runs local ML attribution and passive link inspection to identify risks.
      </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Display conversation history
    for item in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.markdown(f'<div class="user-msg-box">{item["text"]}</div>', unsafe_allow_html=True)

        # Assistant response
        with st.chat_message("assistant"):
            st.markdown(f"""
            <div class="status-pill {item['status_code']}">
              <span>{item['status_label']}</span>
              <span>•</span>
              <span>Risk: {item['risk_score']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

            st.write(item["summary_text"])

            # Highlighted Message Box
            st.markdown('<div class="section-title">Flagged Phrase Analysis</div>', unsafe_allow_html=True)
            st.markdown(item["highlighted_html"], unsafe_allow_html=True)

            # Two Columns: Evidence Breakdown & Next Steps
            col_ev, col_act = st.columns([1.3, 1])

            with col_ev:
                st.markdown('<div class="section-title">Evidence Breakdown</div>', unsafe_allow_html=True)
                evidence_list = []
                for feat in item["ml_result"]["scam_contributions"][:4]:
                    evidence_list.append({
                        "Phrase": f"'{feat['feature']}'",
                        "Source": "ML Token",
                        "Weight": f"+{feat['contribution']:.2f}",
                        "Note": "Statistically correlates with scams"
                    })
                for rule in item["rule_findings"]:
                    matched_words = ", ".join([f"'{m['phrase']}'" for m in rule["matches"]])
                    evidence_list.append({
                        "Phrase": matched_words,
                        "Source": "Rule Engine",
                        "Weight": "Alert",
                        "Note": rule["category"]
                    })
                for u_an in item["url_analyses"]:
                    for flag in u_an["risk_flags"]:
                        evidence_list.append({
                            "Phrase": u_an["hostname"],
                            "Source": "URL Parser",
                            "Weight": "Flag",
                            "Note": flag["label"]
                        })

                if evidence_list:
                    st.dataframe(pd.DataFrame(evidence_list), use_container_width=True, hide_index=True)
                else:
                    st.caption("No suspicious tokens or patterns identified.")

            with col_act:
                st.markdown('<div class="section-title">Recommended Action</div>', unsafe_allow_html=True)
                if item["status_code"] == "high-risk":
                    st.error(
                        "• **Do not click links** or reply.\n"
                        "• **Do not share passwords or OTPs.**\n"
                        "• Verify independently via official numbers."
                    )
                elif item["status_code"] == "medium-risk":
                    st.warning(
                        "• Verify the sender identity.\n"
                        "• Do not enter credentials via unverified links."
                    )
                else:
                    st.success(
                        "• Message appears routine.\n"
                        "• Normal caution applies."
                    )

            # URL Details if present
            if item["urls"]:
                with st.expander(f"Extracted Links ({len(item['urls'])})"):
                    st.caption("Zero outbound network requests made. Evaluated offline.")
                    u_rows = []
                    for a in item["url_analyses"]:
                        u_rows.append({
                            "Link": a["raw_url"],
                            "Host": a["hostname"],
                            "Protocol": a["scheme"].upper(),
                            "Risk Flags": "; ".join([f["label"] for f in a["risk_flags"]]) if a["risk_flags"] else "Clean"
                        })
                    st.table(pd.DataFrame(u_rows))

# ChatGPT-Style Chat Input docked at bottom
prompt = st.chat_input("Paste an SMS or email message here and press Enter...")
if prompt:
    result = analyze_message_content(prompt)
    st.session_state.chat_history.append(result)
    st.rerun()
