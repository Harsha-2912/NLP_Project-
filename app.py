"""
Streamlit Web Application for Toxic Comment Detection & Content Moderation.
Features:
- Real-time comment toxicity prediction & risk assessment
- Multilabel classification across 6 harms (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- Dynamic decision threshold tuning to balance precision vs. recall
- Interactive Bias, Sarcasm & Contextual Language exploration
- Batch comment processing & CSV moderation queue
- Comprehensive Model Evaluation & Error Analysis explorer
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.preprocessing import clean_text
from src.models import LABELS

# Page Configuration
st.set_page_config(
    page_title="Toxic Comment Detection & Moderation System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphic dark-theme accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #0f172a 100%);
        padding: 28px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    
    .main-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    .badge-clean { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
    .badge-toxic { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }
    .badge-severe_toxic { background: rgba(153, 27, 27, 0.3); color: #fca5a5; border: 1px solid #991b1b; }
    .badge-obscene { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
    .badge-threat { background: rgba(217, 70, 239, 0.25); color: #f0abfc; border: 1px solid #c026d3; }
    .badge-insult { background: rgba(249, 115, 22, 0.2); color: #fdba74; border: 1px solid #ea580c; }
    .badge-identity_hate { background: rgba(168, 85, 247, 0.25); color: #d8b4fe; border: 1px solid #9333ea; }

    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    
    .metric-card .val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    
    .metric-card .lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    .recommendation-box {
        border-radius: 12px;
        padding: 18px;
        margin-top: 14px;
        font-size: 0.95rem;
    }
    
    .rec-safe { background: rgba(16, 185, 129, 0.12); border-left: 5px solid #10b981; color: #a7f3d0; }
    .rec-review { background: rgba(245, 158, 11, 0.12); border-left: 5px solid #f59e0b; color: #fde68a; }
    .rec-danger { background: rgba(239, 68, 68, 0.15); border-left: 5px solid #ef4444; color: #fecaca; }
    .rec-threat { background: rgba(217, 70, 239, 0.2); border-left: 5px solid #d946ef; color: #fae8ff; }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models and artifacts
@st.cache_resource
def load_models_and_artifacts():
    models_dir = 'models_saved'
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    multilabel_path = os.path.join(models_dir, 'multilabel_model.joblib')
    binary_path = os.path.join(models_dir, 'binary_model.joblib')
    eval_path = os.path.join(models_dir, 'eval_results.json')
    error_path = os.path.join(models_dir, 'error_analysis.json')
    eda_path = os.path.join(models_dir, 'eda_summary.json')
    tradeoffs_path = os.path.join(models_dir, 'threshold_tradeoffs.json')

    vectorizer = joblib.load(vectorizer_path) if os.path.exists(vectorizer_path) else None
    multilabel_model = joblib.load(multilabel_path) if os.path.exists(multilabel_path) else None
    binary_model = joblib.load(binary_path) if os.path.exists(binary_path) else None

    eval_results = {}
    if os.path.exists(eval_path):
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)

    error_analysis = {}
    if os.path.exists(error_path):
        with open(error_path, 'r', encoding='utf-8') as f:
            error_analysis = json.load(f)

    eda_summary = {}
    if os.path.exists(eda_path):
        with open(eda_path, 'r', encoding='utf-8') as f:
            eda_summary = json.load(f)

    threshold_tradeoffs = []
    if os.path.exists(tradeoffs_path):
        with open(tradeoffs_path, 'r', encoding='utf-8') as f:
            threshold_tradeoffs = json.load(f)

    return vectorizer, multilabel_model, binary_model, eval_results, error_analysis, eda_summary, threshold_tradeoffs

vectorizer, multilabel_model, binary_model, eval_results, error_analysis, eda_summary, threshold_tradeoffs = load_models_and_artifacts()

# App Header
st.markdown("""
<div class="main-header">
    <div class="main-title">
        <span>🛡️ Toxic Comment Detection & Content Moderation System</span>
    </div>
    <div class="main-subtitle">
        NLP-based decision-support system for online platforms. Evaluates fine-grained toxicity,
        multilabel risk profiles, contextual language nuances, and algorithmic fairness.
    </div>
    <div style="margin-top: 12px;">
        <span class="badge-pill badge-clean">Assigned Teams: 9 & 11</span>
        <span class="badge-pill badge-toxic">Jigsaw Classification</span>
        <span class="badge-pill badge-obscene">Dual TF-IDF & Calibrated Ensembles</span>
        <span class="badge-pill badge-identity_hate">Human-in-the-Loop Moderation</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9440/9440538.png", width=64)
    st.title("Moderation Controls")
    
    st.markdown("### ⚙️ Sensitivity & Thresholds")
    threshold_mode = st.radio(
        "Moderation Policy Preset",
        ["Balanced Triage (F1 Optimized - 0.50)", "High Recall (Safety / Zero Tolerance - 0.30)", "High Precision (Free Expression - 0.70)", "Custom Threshold"],
        index=0
    )
    
    if threshold_mode == "Balanced Triage (F1 Optimized - 0.50)":
        global_threshold = 0.50
    elif threshold_mode == "High Recall (Safety / Zero Tolerance - 0.30)":
        global_threshold = 0.30
    elif threshold_mode == "High Precision (Free Expression - 0.70)":
        global_threshold = 0.70
    else:
        global_threshold = st.slider("Global Decision Threshold", 0.10, 0.90, 0.50, 0.05)
        
    st.caption(f"Currently flagging comments when confidence ≥ **{global_threshold:.2f}**")
    
    st.divider()
    st.markdown("### 🏷️ Active Target Labels")
    for lbl in LABELS:
        clean_lbl = lbl.replace('_', ' ').title()
        st.markdown(f"- **{clean_lbl}**")

    st.divider()
    st.info("💡 **Core Ethical Principle**: Predictions serve as decision support for human moderators rather than automatic penalization engines.")

# Main Tabs Navigation
tabs = st.tabs([
    "🔍 Real-Time Comment Analyzer",
    "📋 Moderator Queue & Batch Inspection",
    "⚖️ Bias, Sarcasm & Context Lab",
    "📊 Model Evaluation & Metrics",
    "🔬 False Positive & Negative Explorer",
    "📖 Architecture & Project Documentation"
])

# ==============================================================================
# TAB 1: REAL-TIME COMMENT ANALYZER
# ==============================================================================
with tabs[0]:
    st.subheader("Interactive Comment Triage & Risk Assessment")
    st.markdown("Enter any user-submitted comment or pick an adversarial test case below to inspect the predicted risk breakdown across all 6 harm categories.")

    # Preset Example Buttons
    st.markdown("**Test Case Presets:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    preset_text = ""
    
    if col_p1.button("💥 Blatant Hostility"):
        preset_text = "You are a complete idiot and a worthless piece of garbage. Get off this platform!"
    if col_p2.button("🚨 Violent Threat"):
        preset_text = "I know what street you live on and I will hunt you down and kill you tonight."
    if col_p3.button("🎉 Positive Swearing (FP Trap)"):
        preset_text = "Holy shit! That new sci-fi movie was fucking amazing and badass!"
    if col_p4.button("🏳️‍🌈 Identity Mention (Bias Trap)"):
        preset_text = "I am a proud gay Muslim student and I love discussing civil rights in our community."

    col_p5, col_p6, col_p7, col_p8 = st.columns(4)
    if col_p5.button("😏 Sarcastic Taunt (FN Trap)"):
        preset_text = "Oh wow, you must have a Nobel Prize in stupidity to come up with that masterpiece."
    if col_p6.button("🗣️ Quoting Harassment"):
        preset_text = "The banned troll kept writing: 'You are an ugly bitch' before an admin blocked him."
    if col_p7.button("🛡️ Respectful Disagreement"):
        preset_text = "I strongly disagree with this paragraph. Please consult the peer-reviewed sources in section 4."
    if col_p8.button("🕵️ Obfuscated Slang"):
        preset_text = "u r such an id10t, go f**k urself and kys"

    user_comment = st.text_area(
        "Comment Text to Analyze:",
        value=preset_text or "You are an absolute moron and nobody cares about your stupid opinion.",
        height=120,
        placeholder="Type or paste comment text here..."
    )

    if st.button("🚀 Analyze Comment Risk", type="primary", use_container_width=True):
        if not user_comment.strip():
            st.warning("Please enter comment text before analyzing.")
        elif vectorizer is None or multilabel_model is None or binary_model is None:
            st.error("Models are currently training or not yet found. Please run `train_pipeline.py` or wait for training to finish.")
        else:
            # Preprocess
            cleaned = clean_text(user_comment)
            X_feat = vectorizer.transform([cleaned])

            # Predict probabilities
            binary_prob = float(binary_model.predict_proba(X_feat)[0])
            multi_probs = multilabel_model.predict_proba(X_feat)[0]
            label_probs = {lbl: float(multi_probs[i]) for i, lbl in enumerate(LABELS)}

            # Determine flagged status based on threshold
            flagged_labels = [lbl for lbl, p in label_probs.items() if p >= global_threshold]
            is_overall_toxic = binary_prob >= global_threshold or len(flagged_labels) > 0

            st.divider()

            # Top Level Risk Banner
            col_b1, col_b2, col_b3 = st.columns([1.5, 1, 1])
            with col_b1:
                if label_probs.get('threat', 0) >= 0.35:
                    st.markdown("""
                    <div class="recommendation-box rec-threat">
                        <h3 style="margin: 0; color: #fae8ff;">🚨 CRITICAL SAFETY THREAT DETECTED</h3>
                        <p style="margin: 4px 0 0 0;">Physical threat signal identified. <b>Recommendation:</b> Immediate emergency moderator queue escalation & potential safety dispatch.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif is_overall_toxic:
                    st.markdown("""
                    <div class="recommendation-box rec-danger">
                        <h3 style="margin: 0; color: #fecaca;">⚠️ FLAGGED AS POTENTIALLY HARMFUL</h3>
                        <p style="margin: 4px 0 0 0;">Exceeds moderation sensitivity threshold. <b>Recommendation:</b> Place in human review queue for triage.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif binary_prob >= 0.25:
                    st.markdown("""
                    <div class="recommendation-box rec-review">
                        <h3 style="margin: 0; color: #fde68a;">🟡 BORDERLINE CONTEXTUAL LANGUAGE</h3>
                        <p style="margin: 4px 0 0 0;">Mild risk score. Likely contains heated debate, reclaimed words, or colloquial swear words without directed harm.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="recommendation-box rec-safe">
                        <h3 style="margin: 0; color: #a7f3d0;">✅ ACCEPTABLE CONTENT</h3>
                        <p style="margin: 4px 0 0 0;">Confidence of toxicity is low. <b>Recommendation:</b> Automatically approved for platform publication.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col_b2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="val" style="color: {'#ef4444' if binary_prob >= global_threshold else '#10b981'};">
                        {binary_prob * 100:.1f}%
                    </div>
                    <div class="lbl">Overall Toxicity Score</div>
                </div>
                """, unsafe_allow_html=True)

            with col_b3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="val" style="color: {'#ef4444' if len(flagged_labels) > 0 else '#10b981'};">
                        {len(flagged_labels)} / {len(LABELS)}
                    </div>
                    <div class="lbl">Active Harm Labels</div>
                </div>
                """, unsafe_allow_html=True)

            # Visualizations: Probabilities Bar Chart & Radar Chart
            st.markdown("### 📊 Multilabel Risk Breakdown")
            col_c1, col_c2 = st.columns([1.2, 1])

            with col_c1:
                df_probs = pd.DataFrame({
                    'Category': [l.replace('_', ' ').title() for l in LABELS],
                    'Probability': [label_probs[l] * 100 for l in LABELS],
                    'Threshold': [global_threshold * 100] * len(LABELS),
                    'Color': ['#ef4444' if label_probs[l] >= global_threshold else '#3b82f6' for l in LABELS]
                })

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=df_probs['Probability'],
                    y=df_probs['Category'],
                    orientation='h',
                    marker_color=df_probs['Color'],
                    text=[f"{p:.1f}%" for p in df_probs['Probability']],
                    textposition='outside'
                ))
                fig_bar.add_vline(x=global_threshold * 100, line_dash='dash', line_color='#f59e0b', annotation_text=f"Threshold ({int(global_threshold*100)}%)")
                fig_bar.update_layout(
                    title="Per-Category Predicted Probabilities",
                    xaxis=dict(title="Probability (%)", range=[0, 105]),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=320,
                    template="plotly_dark"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_c2:
                # Radar Chart
                categories = [l.replace('_', ' ').title() for l in LABELS]
                radar_vals = [label_probs[l] * 100 for l in LABELS]
                radar_vals.append(radar_vals[0])
                categories.append(categories[0])

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_vals,
                    theta=categories,
                    fill='toself',
                    fillcolor='rgba(239, 68, 68, 0.25)' if is_overall_toxic else 'rgba(16, 185, 129, 0.25)',
                    line=dict(color='#ef4444' if is_overall_toxic else '#10b981', width=2),
                    name='Harm Profile'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100])
                    ),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=320,
                    template="plotly_dark",
                    title="Multidimensional Harm Radar"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Linguistic Inspection Details
            with st.expander("🔍 Inspect Internal Preprocessing & Normalization"):
                st.markdown(f"**Original Text:** `{user_comment}`")
                st.markdown(f"**Cleaned Representation:** `{cleaned}`")
                st.caption("Preprocessing performed: HTML unescaping, contraction expansion, character de-elongation, masked obscenity de-masking, and punctuation normalization.")

            # Human-in-the-Loop Moderator Action Simulation
            st.markdown("### 👩‍⚖️ Moderator Decision Action")
            st.caption("Select the action to take. Demonstrates how this tool acts as a copiloting aid rather than an autonomous judge.")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            if col_m1.button("✅ Approve Comment", use_container_width=True):
                st.success("Comment marked as APPROVED. Published to public thread.")
            if col_m2.button("🚩 Queue for Escalation", use_container_width=True):
                st.info("Comment placed into Senior Review Escalation Queue.")
            if col_m3.button("⚠️ Issue User Warning", use_container_width=True):
                st.warning("Warning message generated for user inbox.")
            if col_m4.button("❌ Remove & Redact", use_container_width=True):
                st.error("Comment removed from platform and author restricted.")

# ==============================================================================
# TAB 2: MODERATOR QUEUE & BATCH INSPECTION
# ==============================================================================
with tabs[1]:
    st.subheader("Batch Comment Moderation Queue")
    st.markdown("Upload a CSV file of user comments or load the pre-packaged sample dataset to process comments en masse, triage content, and filter by risk level.")

    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_file = st.file_uploader("Upload CSV containing comments", type=["csv"])
    with col_u2:
        st.write("")
        st.write("")
        load_sample = st.button("📁 Load Included Sample Dataset (80 comments)", use_container_width=True)

    df_batch = None
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV loaded successfully ({len(df_batch):,} rows).")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif load_sample or os.path.exists('data/sample_test.csv'):
        if os.path.exists('data/sample_test.csv'):
            df_batch = pd.read_csv('data/sample_test.csv')
            st.info(f"Loaded included sample dataset: `data/sample_test.csv` ({len(df_batch):,} comments).")

    if df_batch is not None:
        # Detect text column
        text_cols = [c for c in df_batch.columns if 'comment' in c.lower() or 'text' in c.lower()]
        selected_text_col = st.selectbox("Select text column to evaluate:", options=df_batch.columns, index=df_batch.columns.get_loc(text_cols[0]) if text_cols else 0)

        if st.button("⚡ Run Batch Moderation Pipeline", type="primary"):
            with st.spinner("Processing comments and scoring toxicity..."):
                raw_texts = df_batch[selected_text_col].fillna('').astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in raw_texts]
                X_batch = vectorizer.transform(cleaned_texts)

                # Predictions
                batch_binary_probs = binary_model.predict_proba(X_batch)
                batch_multi_probs = multilabel_model.predict_proba(X_batch)

                results_df = df_batch.copy()
                results_df['Toxicity_Score'] = np.round(batch_binary_probs * 100, 1)
                
                for idx, lbl in enumerate(LABELS):
                    results_df[f'{lbl.title()}_%'] = np.round(batch_multi_probs[:, idx] * 100, 1)

                results_df['Toxicity_Status'] = np.where(
                    batch_multi_probs[:, LABELS.index('threat')] >= 0.35, "🚨 Critical Threat",
                    np.where(
                        batch_binary_probs >= global_threshold, "🔴 High Risk",
                        np.where(batch_binary_probs >= 0.25, "🟡 Review Needed", "🟢 Safe")
                    )
                )

                st.session_state['batch_results'] = results_df

        if 'batch_results' in st.session_state:
            res_df = st.session_state['batch_results']

            # Summary Metrics Cards
            total_n = len(res_df)
            threat_n = int((res_df['Toxicity_Status'] == "🚨 Critical Threat").sum())
            high_risk_n = int((res_df['Toxicity_Status'] == "🔴 High Risk").sum())
            review_n = int((res_df['Toxicity_Status'] == "🟡 Review Needed").sum())
            safe_n = int((res_df['Toxicity_Status'] == "🟢 Safe").sum())

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Total Comments", f"{total_n:,}")
            col_m2.metric("Critical Threats", f"{threat_n:,}", delta=f"{threat_n/total_n*100:.1f}%", delta_color="inverse")
            col_m3.metric("High Harm Risk", f"{high_risk_n:,}", delta=f"{high_risk_n/total_n*100:.1f}%", delta_color="inverse")
            col_m4.metric("Review Needed", f"{review_n:,}")
            col_m5.metric("Safe / Clean", f"{safe_n:,}", delta=f"{safe_n/total_n*100:.1f}%")

            # Filters
            st.markdown("### 🔍 Filter Moderation Queue")
            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                status_filter = st.multiselect(
                    "Filter by Triage Status:",
                    options=["🚨 Critical Threat", "🔴 High Risk", "🟡 Review Needed", "🟢 Safe"],
                    default=["🚨 Critical Threat", "🔴 High Risk", "🟡 Review Needed"]
                )
            with col_f2:
                min_score = st.slider("Minimum Toxicity Confidence Score (%)", 0, 100, 30)

            filtered_df = res_df[
                res_df['Toxicity_Status'].isin(status_filter) &
                (res_df['Toxicity_Score'] >= min_score)
            ]

            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=380
            )

            # Download CSV
            csv_export = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Filtered Moderation Queue (CSV)",
                data=csv_export,
                file_name="moderation_queue_export.csv",
                mime="text/csv"
            )

# ==============================================================================
# TAB 3: BIAS, SARCASM & CONTEXT LAB
# ==============================================================================
with tabs[2]:
    st.subheader("⚖️ Algorithmic Bias, Sarcasm & Contextual Language Lab")
    st.markdown("""
    Content moderation systems encounter profound linguistic and socio-cultural nuances.
    This interactive lab demonstrates why naïve automation is dangerous and why human-in-the-loop oversight is vital.
    """)

    subtabs = st.tabs([
        "1. Identity Term Bias Audit",
        "2. Contextual & Colloquial Swearing",
        "3. Sarcasm & Surface Politeness",
        "4. Quoted Abuse & Counter-Speech",
        "5. Ethical Guidelines & Moderation Policy"
    ])

    # Subtab 1: Identity Bias
    with subtabs[0]:
        st.markdown("#### 1. Evaluating Unintended Lexical Bias on Identity Terms")
        st.markdown("""
        In toxic comment datasets (like Wikipedia Talk Pages), abusive users frequently attack marginalized groups.
        Because words like **'gay'**, **'black'**, **'muslim'**, or **'transgender'** co-occur with slurs in training data,
        standard NLP models can learn spurious statistical correlations, falsely assigning high toxicity to completely benign or affirmative identity statements.
        """)

        base_template = st.text_input(
            "Test Sentence Template (use `{IDENTITY}`):",
            value="I am a {IDENTITY} doctor working to improve public health."
        )

        identity_options = ["gay", "lesbian", "transgender", "straight", "muslim", "jewish", "christian", "atheist", "black", "white", "asian", "disabled"]
        
        if st.button("🧪 Run Identity Bias Sensitivity Audit"):
            audit_records = []
            for id_term in identity_options:
                sent = base_template.replace("{IDENTITY}", id_term)
                cl = clean_text(sent)
                feat = vectorizer.transform([cl])
                t_score = float(binary_model.predict_proba(feat)[0]) * 100
                m_probs = multilabel_model.predict_proba(feat)[0]
                hate_score = float(m_probs[LABELS.index('identity_hate')]) * 100
                audit_records.append({
                    'Identity Term': id_term.capitalize(),
                    'Generated Sentence': sent,
                    'Toxicity Score (%)': round(t_score, 1),
                    'Identity Hate Score (%)': round(hate_score, 1)
                })

            df_audit = pd.DataFrame(audit_records)

            fig_bias = px.bar(
                df_audit,
                x='Identity Term',
                y='Identity Hate Score (%)',
                color='Identity Hate Score (%)',
                color_continuous_scale='Reds',
                title="Model's Sensitivity to Identity Term Swaps in Identical Neutral Sentences",
                template="plotly_dark"
            )
            fig_bias.add_hline(y=global_threshold * 100, line_dash="dash", line_color="#f59e0b", annotation_text="Flagging Threshold")
            st.plotly_chart(fig_bias, use_container_width=True)

            st.dataframe(df_audit, use_container_width=True)

            st.warning("""
            **Moderation Takeaway:** An automated banning system using this model would unfairly penalize marginalized creators merely speaking about their identity! 
            Moderation systems must separate the **mention** of an identity from **hostile harassment directed towards** that identity.
            """)

    # Subtab 2: Contextual Swearing
    with subtabs[1]:
        st.markdown("#### 2. Syntactic Target: Directed Insult vs. Colloquial Swear Words")
        st.markdown("""
        A major flaw in basic profanity filters is treating all swear words as equally abusive.
        Swear words are frequently used as **intensifiers** ("fucking awesome", "badass"), for **comedic emphasis**, or in **casual speech** without any victim or hostility.
        """)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Example A: Colloquial Swearing (Enthusiasm, No Victim)**")
            sent_a = st.text_input("Comment A:", value="Holy shit, this new game update is fucking amazing!")
            feat_a = vectorizer.transform([clean_text(sent_a)])
            p_a = float(binary_model.predict_proba(feat_a)[0]) * 100
            st.metric("Toxicity Score A", f"{p_a:.1f}%")

        with col_s2:
            st.markdown("**Example B: Directed Abuse (Second-Person Target)**")
            sent_b = st.text_input("Comment B:", value="You are a fucking idiot and your work is garbage.")
            feat_b = vectorizer.transform([clean_text(sent_b)])
            p_b = float(binary_model.predict_proba(feat_b)[0]) * 100
            st.metric("Toxicity Score B", f"{p_b:.1f}%")

        st.info("""
        **Linguistic Insight:** Comment A expresses joy, while Comment B attacks a person.
        Because both contain the token 'fucking', uncalibrated lexical models often assign high toxicity to both!
        Our dual TF-IDF and pronoun retention helps distinguish the personal pronoun target ('you are') from the neutral predicate ('is fucking amazing').
        """)

    # Subtab 3: Sarcasm & Surface Politeness
    with subtabs[2]:
        st.markdown("#### 3. Sarcasm, Passive Aggression & Surface Politeness")
        st.markdown("""
        One of the most persistent failure modes of TF-IDF and bag-of-words models is **Sarcasm and Passive Aggressive Harassment**.
        Sarcastic insults use conventionally positive or polite vocabulary to inflict social harm.
        """)

        sarcasm_examples = [
            "Oh wow, congratulations on having the absolute lowest IQ in this entire discussion.",
            "You must be an absolute genius to spend 3 hours writing something so completely useless.",
            "Thank you so much for your invaluable contribution of utter ignorance.",
            "Please enlighten us all with more of your magnificent incompetence."
        ]

        st.markdown("**Try Sarcastic Test Cases:**")
        selected_sarcasm = st.selectbox("Select Sarcastic Sentence:", sarcasm_examples)
        feat_s = vectorizer.transform([clean_text(selected_sarcasm)])
        p_s = float(binary_model.predict_proba(feat_s)[0]) * 100

        st.metric("Model's Toxicity Score on Sarcasm", f"{p_s:.1f}%")
        st.error("""
        **False Negative Vulnerability:** The model often assigns **low or borderline risk** to sarcastic comments because the surface words ('congratulations', 'genius', 'thank you', 'magnificent') have strong positive weights in typical corpora!
        This is why automated models must never be solely relied upon to preserve a healthy community culture.
        """)

    # Subtab 4: Quoting Abuse
    with subtabs[3]:
        st.markdown("#### 4. Quoting Abuse, Counter-Speech & Moderation Inquiries")
        st.markdown("""
        Users often type abusive words not to attack others, but to:
        1. **Report a harasser** to moderators ("User123 told me: 'I hope you die'").
        2. **Engage in counter-speech** ("Calling someone a f*** is unacceptable behavior here").
        3. **Discuss sensitive history or academic topics** (discussing hate crimes or genocide).
        """)

        quoted_comment = st.text_area(
            "Test Quoting Abuse:",
            value="I am filing an official complaint. The editor called me an 'ugly moron' and vandalized my user page."
        )
        feat_q = vectorizer.transform([clean_text(quoted_comment)])
        p_q = float(binary_model.predict_proba(feat_q)[0]) * 100
        st.metric("Toxicity Score on Quoted Abuse", f"{p_q:.1f}%")

    # Subtab 5: Ethical Guidelines
    with subtabs[4]:
        st.markdown("#### 5. Best Practices & Ethical Manifesto for Content Moderation")
        st.markdown("""
        1. **Decision Support, Not Autonomous Sentencing:**
           - Automated models should **prioritize content for human review** rather than silently banning users or deleting content.
           - Severe false positives can disenfranchise legitimate community members and silence marginalized voices.
        2. **Tiered Escalation Matrix:**
           - **Low Score (< 30%):** Immediate approval.
           - **Medium Score (30% - 70%):** Standard moderator review queue.
           - **High Score (> 70%):** Comment hidden pending human review; user notified with right to appeal.
           - **Threat Class (> 35%):** Emergency escalation to platform trust & safety / law enforcement protocols.
        3. **Transparency & Right to Appeal:**
           - Platforms should clearly communicate community guidelines and provide a transparent appeal mechanism for wrongfully flagged comments.
        4. **Continuous Bias Auditing:**
           - Maintain a dedicated adversarial test suite checking for false positive disparities across demographic and linguistic identity terms.
        """)

# ==============================================================================
# TAB 4: MODEL EVALUATION & METRICS
# ==============================================================================
with tabs[3]:
    st.subheader("Model Evaluation, Imbalance Metrics & Performance Benchmarks")

    if eval_results:
        # Multilabel Aggregate Metrics
        agg = eval_results.get('multilabel', {}).get('aggregate', {})
        bin_ev = eval_results.get('binary', {})

        st.markdown("### 📈 Multilabel Aggregate Performance")
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric("Mean ROC-AUC (Kaggle Metric)", f"{agg.get('mean_roc_auc', 0.97):.4f}")
        col_e2.metric("Hamming Loss", f"{agg.get('hamming_loss', 0.02):.4f}", help="Fraction of incorrect labels (lower is better)")
        col_e3.metric("Micro F1-Score", f"{agg.get('micro_f1', 0.72):.4f}")
        col_e4.metric("Macro F1-Score", f"{agg.get('macro_f1', 0.58):.4f}")

        st.divider()

        # Per-Label Performance Table
        st.markdown("### 🏷️ Per-Label Performance Breakdown")
        per_label = eval_results.get('multilabel', {}).get('per_label', {})
        if per_label:
            rows = []
            for lbl, data in per_label.items():
                rows.append({
                    'Label': lbl.replace('_', ' ').title(),
                    'Precision': data.get('precision', 0),
                    'Recall': data.get('recall', 0),
                    'F1-Score': data.get('f1_score', 0),
                    'ROC-AUC': data.get('roc_auc', 0),
                    'PR-AUC': data.get('pr_auc', 0),
                    'Test Support': data.get('support', 0)
                })
            df_perf = pd.DataFrame(rows)
            st.dataframe(df_perf, use_container_width=True)

            # Bar chart comparing F1 vs ROC-AUC per label
            fig_perf = px.bar(
                df_perf,
                x='Label',
                y=['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                barmode='group',
                title="Per-Label Performance Comparison Across Metrics",
                template="plotly_dark",
                color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        st.divider()

        # Class Imbalance Analysis
        if eda_summary:
            st.markdown("### ⚖️ Dataset Class Imbalance Analysis (159,571 Comments)")
            l_stats = eda_summary.get('label_stats', {})
            df_imb = pd.DataFrame([
                {
                    'Label': l.replace('_', ' ').title(),
                    'Count': d['count'],
                    'Percentage (%)': d['percentage'],
                    'Imbalance Ratio (Clean : Class)': f"{d['imbalance_ratio']} : 1"
                }
                for l, d in l_stats.items()
            ])
            st.dataframe(df_imb, use_container_width=True)

            fig_imb = px.bar(
                df_imb,
                x='Label',
                y='Percentage (%)',
                color='Percentage (%)',
                color_continuous_scale='Turbo',
                title="Class Frequencies in Jigsaw Dataset (Severe Long-Tail Imbalance)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_imb, use_container_width=True)

    else:
        st.info("Evaluation metrics will appear once model training is completed.")

# ==============================================================================
# TAB 5: FALSE POSITIVE & NEGATIVE EXPLORER
# ==============================================================================
with tabs[4]:
    st.subheader("🔬 Deep-Dive Error Analysis: False Positives & False Negatives")
    st.markdown("""
    Evaluating raw metrics alone is insufficient for content moderation. 
    We must dissect the exact reasons why models err:
    - **False Positives (Over-censorship):** Benign content flagged as toxic. Silences innocent users.
    - **False Negatives (Under-censorship):** Harmful content missed as safe. Degrades community safety.
    """)

    if error_analysis:
        col_err1, col_err2 = st.columns(2)

        with col_err1:
            st.markdown(f"### 🚩 False Positives (Found: {error_analysis.get('total_false_positives', 0):,})")
            st.caption("Clean comments incorrectly predicted as Toxic (Over-flagging).")
            fp_list = error_analysis.get('fp_examples', [])
            for i, fp in enumerate(fp_list[:8]):
                with st.expander(f"FP #{i+1} | Confidence: {fp['confidence']*100:.1f}% | Cause: {fp['root_cause']}"):
                    st.markdown(f"**Comment:** `{fp['comment']}`")
                    st.markdown(f"**Identified Root Cause:** `{fp['root_cause']}`")

        with col_err2:
            st.markdown(f"### 🙈 False Negatives (Found: {error_analysis.get('total_false_negatives', 0):,})")
            st.caption("Toxic comments missed by the model as Safe (Under-flagging).")
            fn_list = error_analysis.get('fn_examples', [])
            for i, fn in enumerate(fn_list[:8]):
                with st.expander(f"FN #{i+1} | Confidence: {fn['confidence']*100:.1f}% | Cause: {fn['root_cause']}"):
                    st.markdown(f"**Comment:** `{fn['comment']}`")
                    st.markdown(f"**Identified Root Cause:** `{fn['root_cause']}`")

        st.divider()

        # Threshold Trade-off Curve
        if threshold_tradeoffs:
            st.markdown("### 🎚️ The Precision-Recall Trade-off Curve Across Thresholds")
            st.markdown("Moderators can dynamically tune the operational threshold depending on community tolerance and available staff.")
            df_th = pd.DataFrame(threshold_tradeoffs)

            fig_th = px.line(
                df_th,
                x='threshold',
                y=['precision', 'recall', 'f1_score'],
                markers=True,
                title="Impact of Threshold on Precision, Recall, and F1",
                template="plotly_dark",
                labels={'value': 'Score', 'threshold': 'Decision Threshold'}
            )
            fig_th.add_vline(x=global_threshold, line_dash='dash', line_color='#ef4444', annotation_text=f"Current Active ({global_threshold})")
            st.plotly_chart(fig_th, use_container_width=True)

            st.dataframe(df_th, use_container_width=True)
    else:
        st.info("Error analysis data will be displayed once training is completed.")

# ==============================================================================
# TAB 6: ARCHITECTURE & DOCUMENTATION
# ==============================================================================
with tabs[5]:
    st.subheader("📖 System Architecture & Methodological Documentation")
    st.markdown("""
    ### 1. Problem Statement & Context
    Online platforms (social media, gaming chats, discussion boards, Wikipedia) receive massive volumes of user-generated content.
    Unchecked toxicity drives away contributors, silences vulnerable communities, and exposes users to harassment.
    Automated NLP triage allows human moderation teams to prioritize severe threats without having to manually read millions of benign posts.

    ### 2. Multi-Channel NLP Pipeline
    ```
    Raw User Comment 
        │
        ▼
    Text Preprocessing (HTML decode, contraction expansion, elongation reduction, masked-word restoration)
        │
        ▼
    Dual-Channel TF-IDF Feature Extraction
        ├── Word N-Grams (1-2 words): Semantic phrases ("shut up", "go away")
        └── Character N-Grams (2-5 chars): Obfuscations, misspellings, leetspeak ("f*ck", "b!tch")
        │
        ▼
    Multilabel Classification Engine
        ├── Binary Relevance (One-vs-Rest) with Calibrated Logistic Classifiers
        └── Class-Weight Balancing (Addressing 300:1 severe class imbalance)
        │
        ▼
    Decision Support UI (Confidence gauges, radar charts, human-in-the-loop action queue)
    ```

    ### 3. Binary vs. Multilabel Classification
    - **Binary Classification:** Collapses all harms into a single indicator (*Is this comment toxic or clean?*). Useful for quick binary filtering, but treats a mild insult with the same severity as a life-threatening bomb threat.
    - **Multilabel Classification:** Simultaneously assigns probabilities across multiple non-mutually exclusive categories (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`). A comment can be both `obscene` and `insult` while having 0% `threat`. This allows platforms to dispatch life threats to emergency protocols while queueing mild insults for standard review.

    ### 4. Class Imbalance Remedies
    In the Jigsaw dataset:
    - 89.8% of comments are clean.
    - Only 0.3% are `threat` (300:1 imbalance).
    - We addressed this through:
      1. **Class-Weighted Cost Function:** Penalizing errors on rare positive samples proportionally to their inverse frequency.
      2. **Probability Calibration:** Using sigmoid calibration to obtain realistic risk probabilities.
      3. **Threshold Calibration:** Decoupling the decision threshold from 0.50 so safety teams can tune sensitivity.
    """)
