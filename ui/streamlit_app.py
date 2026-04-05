# ui/streamlit_app.py
import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.predict import LegalDocumentClassifier

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Document Classifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f0f;
    color: #e8e4dc;
}

.main { background-color: #0f0f0f; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }

h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem !important;
    color: #e8e4dc;
    letter-spacing: -0.02em;
    margin-bottom: 0 !important;
}

h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #e8e4dc;
}

.subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #8a8070;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.stTextArea textarea {
    background-color: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 4px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stTextArea textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 1px #c9a84c !important;
}

.stButton > button {
    background-color: #c9a84c !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
}

.stButton > button:hover {
    background-color: #e0bc60 !important;
}

.result-card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-left: 3px solid #c9a84c;
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #8a8070;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.result-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #e8e4dc;
}

.result-desc {
    font-size: 0.85rem;
    color: #8a8070;
    margin-top: 0.2rem;
}

.confidence-badge {
    display: inline-block;
    background: #c9a84c22;
    border: 1px solid #c9a84c55;
    color: #c9a84c;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.7rem;
    border-radius: 2px;
    margin-top: 0.5rem;
}

.score-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    gap: 0.8rem;
}

.score-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #8a8070;
    width: 100px;
    flex-shrink: 0;
}

.score-bar-bg {
    flex: 1;
    height: 4px;
    background: #2e2e2e;
    border-radius: 2px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    background: #c9a84c;
    border-radius: 2px;
}

.score-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #5a5040;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

.info-card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}

.info-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #c9a84c;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.info-item {
    font-size: 0.82rem;
    color: #8a8070;
    margin-bottom: 0.2rem;
}

.divider {
    border: none;
    border-top: 1px solid #2e2e2e;
    margin: 1.5rem 0;
}

.stRadio > label { color: #8a8070 !important; font-size: 0.85rem !important; }
.stRadio > div { gap: 1rem !important; }

.stFileUploader {
    background: #1a1a1a !important;
    border: 1px dashed #2e2e2e !important;
    border-radius: 4px !important;
}

.stSpinner > div { border-top-color: #c9a84c !important; }

.stat-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.8rem;
}

.stat-box {
    flex: 1;
    background: #111;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    padding: 0.8rem;
    text-align: center;
}

.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #c9a84c;
}

.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #5a5040;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    return LegalDocumentClassifier()


# ── Helpers ────────────────────────────────────────────────────────────────────
def render_score_bars(all_scores: dict):
    bars_html = ""
    for label, score in all_scores.items():
        pct = score * 100
        bars_html += f"""
        <div class="score-row">
            <span class="score-label">{label}</span>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{pct:.1f}%"></div>
            </div>
            <span class="score-pct">{pct:.1f}%</span>
        </div>
        """
    return bars_html


def render_result(result: dict, text: str):
    confidence_pct = result["confidence"] * 100
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Document Type</div>
        <div class="result-value">{result['predicted_class'].upper()}</div>
        <div class="result-desc">{result['description']}</div>
        <div class="confidence-badge">{confidence_pct:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("<div class='result-label' style='margin-bottom:0.8rem'>All Class Scores</div>",
                unsafe_allow_html=True)
    st.markdown(render_score_bars(result["all_scores"]), unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    words = len(text.split())
    chars = len(text)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box">
            <div class="stat-num">{words:,}</div>
            <div class="stat-lbl">Words</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">{chars:,}</div>
            <div class="stat-lbl">Chars</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">{confidence_pct:.0f}%</div>
            <div class="stat-lbl">Confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_info():
    st.markdown("""
    <div class="info-card">
        <div class="info-title">Model</div>
        <div class="info-item">nlpaueb/legal-bert-base-uncased</div>
        <div class="info-item">Fine-tuned on LexGLUE benchmark</div>
    </div>
    <div class="info-card">
        <div class="info-title">Performance</div>
        <div class="info-item">85% Test Accuracy</div>
        <div class="info-item">84.66% Macro F1</div>
        <div class="info-item">35,000 training samples</div>
    </div>
    <div class="info-card">
        <div class="info-title">Document Types</div>
        <div class="info-item">SCOTUS — Supreme Court Rulings</div>
        <div class="info-item">LEDGAR — Contract Clauses</div>
        <div class="info-item">EUR-Lex — EU Regulations</div>
        <div class="info-item">ECtHR A/B — Human Rights Cases</div>
        <div class="info-item">CaseHOLD — Legal Precedents</div>
        <div class="info-item">UnfairToS — ToS Clauses</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("<h1>Legal Document Classifier</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Legal-BERT · LexGLUE · 7 Document Categories · 85% Accuracy</div>",
        unsafe_allow_html=True
    )

    # Load model
    try:
        classifier = load_classifier()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Layout
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        input_method = st.radio(
            "Input method",
            ["Paste Text", "Upload File"],
            horizontal=True,
            label_visibility="collapsed"
        )

        user_text = ""

        if input_method == "Paste Text":
            user_text = st.text_area(
                "Document text",
                height=320,
                placeholder="Paste your legal document text here...\n\nWorks best with: court opinions, contract clauses, EU regulations, terms of service.",
                label_visibility="collapsed"
            )
        else:
            uploaded = st.file_uploader(
                "Upload file",
                type=["txt"],
                label_visibility="collapsed",
                help="Upload a .txt file containing your legal document"
            )
            if uploaded:
                user_text = uploaded.read().decode("utf-8")
                st.text_area(
                    "File preview",
                    value=user_text[:1000] + ("..." if len(user_text) > 1000 else ""),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )

        classify_clicked = st.button("CLASSIFY DOCUMENT", type="primary")

    with col2:
        if classify_clicked:
            if not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                with st.spinner("Analysing..."):
                    try:
                        result = classifier.predict(user_text)
                        render_result(result, user_text)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
        else:
            render_sidebar_info()


if __name__ == "__main__":
    main()