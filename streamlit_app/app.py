import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from model_loader import ChestXRayModel

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Multi-Label Chest X-Ray Disease Classifier",
    layout="wide"
)

# =============================================================================
# THEME TOGGLE
# =============================================================================
dark_mode = st.toggle("Dark Mode", value=False)

bg = "#0b1220" if dark_mode else "#f8fafc"
text = "#e5e7eb" if dark_mode else "#0f172a"

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown(f"""
<style>
.stApp {{
    background-color: {bg};
    color: {text};
}}

/* ================= HEADER ================= */
.header-card {{
    background: linear-gradient(135deg, #2563eb, #1e40af);
    padding: 24px 28px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 24px;
}}

.header-title {{
    font-size: 2.4rem;
    font-weight: 700;
    color: white;
}}

.header-subtitle {{
    font-size: 1rem;
    color: #dbeafe;
    margin-top: 6px;
}}

/* ================= DISEASE BADGE ================= */
.disease-badge {{
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #ffffff;
    padding: 16px 24px;
    border-radius: 14px;
    margin: 12px 0;
    font-weight: 700;
    font-size: 1.3rem;
    min-width: 340px;
    text-align: center;
    letter-spacing: 0.5px;
    box-shadow: 0 6px 16px rgba(34, 197, 94, 0.35);
}}

/* ================= INFO BOX ================= */
.info-box {{
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: #ffffff;
    padding: 16px 22px;
    border-radius: 14px;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 12px;
    text-align: center;
    box-shadow: 0 6px 16px rgba(14, 165, 233, 0.35);
}}

/* ================= BUTTONS ================= */
.stButton > button {{
    background-color: #2563eb;
    color: white;
    border-radius: 14px;
    font-weight: 600;
    padding: 12px 18px;
    border: none;
}}

.stButton > button:hover {{
    background-color: #1e40af;
}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL
# =============================================================================
@st.cache_resource
def load_model():
    return ChestXRayModel(
        model_path="final_densenet121_model.h5",
        metadata_path="model_metadata.json",
        thresholds_path="optimal_thresholds.npy",
        class_weights_path="class_weights.npy"
    )

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():

    # ---------------- HEADER ----------------
    st.markdown("""
    <div class="header-card">
        <div class="header-title">Multi-Label Chest X-Ray Disease Classifier</div>
        <div class="header-subtitle">
             Academic & Research Use Only
        </div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()

    col_left, col_right = st.columns([1.2, 1.8], gap="large")

    # -------------------------------------------------------------------------
    # LEFT COLUMN: IMAGE UPLOAD
    # -------------------------------------------------------------------------
    with col_left:
        st.subheader("Image Upload")

        uploaded_file = st.file_uploader(
            "Upload Chest X-ray Image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded_file is None:
            st.markdown(
                "<div class='info-box'>Upload a chest X-ray image to begin analysis</div>",
                unsafe_allow_html=True
            )
            return

        image_bytes = uploaded_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Invalid image file")
            return

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            width="stretch"
        )

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: ANALYSIS RESULTS
    # -------------------------------------------------------------------------
    with col_right:
        st.subheader("Analysis Results")

        with st.spinner("Analyzing image..."):
            detected, _, all_scores = model.predict(
                image,
                return_all_scores=True
            )

        if detected:
            st.markdown(
                "<h3 style='color:#22c55e; font-weight:700;'>Detected Abnormal Findings</h3>",
                unsafe_allow_html=True
            )

            for d in detected:
                st.markdown(
                    f"<div class='disease-badge'>{d['disease']}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div class='info-box'>No abnormal findings detected</div>",
                unsafe_allow_html=True
            )

        st.session_state.all_scores = all_scores
        st.session_state.thresholds = model.thresholds
        st.session_state.class_names = model.class_names

    # -------------------------------------------------------------------------
    # NAVIGATION
    # -------------------------------------------------------------------------
    st.divider()
    col1, col2, _ = st.columns([1, 1, 2])

    if col1.button("Detailed Results", use_container_width=True):
        st.session_state.view = "details"

    if col2.button("Model Scores", use_container_width=True):
        st.session_state.view = "scores"

    # -------------------------------------------------------------------------
    # DETAILED RESULTS TABLE
    # -------------------------------------------------------------------------
    if st.session_state.get("view") == "details":
        st.markdown("### Detailed Results")

        rows = []
        for i, name in enumerate(st.session_state.class_names):
            score = st.session_state.all_scores[name]
            threshold = st.session_state.thresholds[i]

            rows.append({
                "Disease": name,
                "Score": round(score, 4),
                "Threshold": round(threshold, 4),
                "Status": "YES" if score >= threshold else "NO"
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # MODEL SCORE VISUALIZATION
    # -------------------------------------------------------------------------
    if st.session_state.get("view") == "scores":
        st.markdown("### Model Output Scores")

        fig, ax = plt.subplots(figsize=(12, 5))
        diseases = list(st.session_state.all_scores.keys())
        scores = list(st.session_state.all_scores.values())
        thresholds = st.session_state.thresholds

        x = np.arange(len(diseases))
        colors = [
            "#22c55e" if scores[i] >= thresholds[i] else "#9ca3af"
            for i in range(len(scores))
        ]

        ax.bar(x, scores, color=colors)
        ax.plot(x, thresholds, linestyle="--", linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(diseases, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability Score")
        ax.set_title("Score vs Threshold Comparison")
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)

# =============================================================================
# SESSION STATE INIT
# =============================================================================
if "view" not in st.session_state:
    st.session_state.view = None

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()
