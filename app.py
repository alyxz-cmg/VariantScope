"""Streamlit demo for the Variant Impact Predictor."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from pipeline import run_pipeline
from feature_engineering import extract_features_from_string, parse_variant
from evaluation import get_roc_curves, evaluate_model, compute_shap_values

st.set_page_config(page_title="Variant Impact Predictor", layout="wide")
st.title("Variant Impact Predictor for Missense Mutations")



@st.cache_resource(show_spinner="Training models (one-time)...")
def load_pipeline():
    cfg = Config()
    models, results, X_test, y_test = run_pipeline(cfg)
    return cfg, models, results, X_test, y_test


cfg, models, results, X_test, y_test = load_pipeline()


st.sidebar.header("⚙️ Controls")
selected_model_name = st.sidebar.selectbox(
    "Primary model for predictions",
    [m.name for m in models],
    index=1,
)
selected_model = next(m for m in models if m.name == selected_model_name)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Synthetic (mimics ClinVar)")
st.sidebar.markdown(f"**Test set size:** {len(X_test)}")
st.sidebar.markdown(f"**Features:** {len(cfg.feature_names)}")


tab_predict, tab_compare, tab_explore = st.tabs(
    ["🔍 Predict", "📊 Model Comparison", "📈 Data Explorer"]
)

with tab_predict:
    st.subheader("Enter a missense variant")
    col1, col2 = st.columns([2, 1])
    with col1:
        variant_input = st.text_input(
            "Variant (e.g. A123T, p.Ala123Thr)", value="G56D"
        )
    with col2:
        conservation = st.slider("Conservation score", 0.0, 1.0, 0.7, 0.05)
        in_domain = st.checkbox("In functional domain?", value=True)

    if st.button("Predict", type="primary"):
        try:
            ref, pos, alt = parse_variant(variant_input)
            feats = extract_features_from_string(
                variant_input, conservation_score=conservation, in_domain=in_domain
            )
            X_single = np.array([[feats[f] for f in cfg.feature_names]], dtype=np.float32)

            st.markdown(f"**Variant:** {ref}{pos}{alt}")
            st.markdown("---")

            pred_cols = st.columns(len(models))
            for i, m in enumerate(models):
                prob = float(m.predict_proba(X_single)[0])
                label = "🔴 Pathogenic" if prob >= 0.5 else "🟢 Benign"
                with pred_cols[i]:
                    st.metric(m.name, label, f"{prob:.1%} pathogenic")

            st.markdown("---")
            st.subheader(f"Feature Importance ({selected_model_name})")
            sv = compute_shap_values(
                selected_model, X_test[:100], X_single, cfg.feature_names
            )
            if sv is not None:
                sv_flat = sv.flatten()
                shap_df = pd.DataFrame({
                    "feature": cfg.feature_names,
                    "shap_value": sv_flat,
                    "abs_shap": np.abs(sv_flat),
                }).sort_values("abs_shap", ascending=True)

                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ["#ff4b4b" if v > 0 else "#4b8bff" for v in shap_df["shap_value"]]
                ax.barh(shap_df["feature"], shap_df["shap_value"], color=colors)
                ax.set_xlabel("SHAP Value (→ pathogenic | ← benign)")
                ax.set_title(f"SHAP Feature Contributions — {selected_model_name}")
                plt.tight_layout()
                st.pyplot(fig)

            with st.expander("View extracted features"):
                st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "Value"}))

        except ValueError as e:
            st.error(str(e))

with tab_compare:
    st.subheader("Test Set Performance")
    comp_df = results["comparison"]
    st.dataframe(comp_df.style.highlight_max(axis=0, color="#d4edda").format("{:.4f}"))

    st.subheader("ROC Curves")
    curves = get_roc_curves(models, X_test, y_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    for mname, c in curves.items():
        auc_val = comp_df.loc[mname, "roc_auc"]
        ax.plot(c["fpr"], c["tpr"], label=f"{mname} (AUC={auc_val:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)