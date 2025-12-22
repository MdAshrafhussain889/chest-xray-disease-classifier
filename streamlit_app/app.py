import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from model_loader import ChestXRayModel
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Multi-Label Chest X-Ray Disease Classifier",
    page_icon=None,
    layout="wide"
)

# Custom CSS for professional appearance
st.markdown("""
    <style>
    /* Center and style the main title */
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0px;
        color: white;
    }

    .subtitle {
        text-align: center;
        font-size: 0.95em;
        margin-top: 5px;
        margin-bottom: 20px;
        color: #e0e0e0;
    }

    .disease-badge {
        background-color: #e7f3ff;
        color: #0066cc;
        padding: 12px 20px;
        border-radius: 4px;
        display: block;
        margin: 8px 0;
        font-weight: 700;
        width: fit-content;
        min-width: 300px;
        text-align: center;
        border: 1px solid #b3d9ff;
    }

    /* Highlight rows with YES status */
    tbody tr:has([data-status="YES"]) {
        background-color: rgba(76, 175, 80, 0.15) !important;
    }

    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL (CACHED)
# ============================================================================
@st.cache_resource
def load_model_cached():
    """Load model once and cache it."""
    try:
        model = ChestXRayModel(
            model_path="final_densenet121_model.h5",
            metadata_path="model_metadata.json",
            thresholds_path="optimal_thresholds.npy",
            class_weights_path="class_weights.npy"
        )
        return model, None
    except Exception as e:
        return None, str(e)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # CENTERED TITLE - BIG AND BOLD
    st.markdown(
        '<div class="main-title">Multi-Label Chest X-Ray Disease Classifier</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Model: Fine-tuned DenseNet121 | Purpose: Academic and research use only</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Load Model
    model, error = load_model_cached()

    if error:
        st.error(f"Failed to load model: {error}")
        st.info("Ensure all model files are in the same directory as app.py")
        return

    # ====================================================================
    # LAYOUT: LEFT COLUMN (IMAGE) | RIGHT COLUMN (ANALYSIS)
    # ====================================================================
    col_upload, col_analysis = st.columns([1.2, 1.8], gap="large")

    # LEFT COLUMN: IMAGE UPLOAD AND DISPLAY
    with col_upload:
        st.subheader("Image Upload")

        st.write("**Supported:** PNG, JPG, JPEG")

        uploaded_file = st.file_uploader(
            "Select X-ray image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded_file is None:
            st.info(" Upload a chest X-ray image")
            return

        # Process image
        image_data = uploaded_file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not process image")
            return

        # Display image
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # RIGHT COLUMN: ANALYSIS RESULTS
    with col_analysis:
        st.subheader("Analysis Results")

        try:
            with st.spinner("Analyzing image..."):
                detected_diseases, raw_scores, all_scores = model.predict(
                    image,
                    return_all_scores=True
                )

            # Display predicted diseases in COLUMN FORMAT with UNIFORM SIZE
            if detected_diseases:
                st.markdown("**Predicted Diseases:**")
                for disease in detected_diseases:
                    st.markdown(
                        f"<div class='disease-badge'>{disease['disease']}</div>", 
                        unsafe_allow_html=True
                    )
            else:
                st.info("No abnormality found")

            st.markdown("")

            # Store results in session state for later use
            st.session_state.detected_diseases = detected_diseases
            st.session_state.all_scores = all_scores
            st.session_state.model_thresholds = model.thresholds if hasattr(model, 'thresholds') else [0.5] * len(model.class_names)
            st.session_state.model_class_names = model.class_names

        except Exception as e:
            st.error(f"Analysis error: {e}")

    # ====================================================================
    # NAVIGATION BUTTONS
    # ====================================================================
    st.divider()

    st.markdown("### View Analysis Details")

    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])

    with col_btn1:
        if st.button("Detailed Results", use_container_width=True, key="btn_results"):
            st.session_state.show_detailed = True
            st.session_state.show_visualization = False

    with col_btn2:
        if st.button("Model Output Scores", use_container_width=True, key="btn_scores"):
            st.session_state.show_visualization = True
            st.session_state.show_detailed = False

    st.divider()

    # ====================================================================
    # DETAILED RESULTS SECTION
    # ====================================================================
    if st.session_state.get("show_detailed", False):
        st.markdown("### Detailed Results (All 14 Classes)")

        try:
            # Create results dataframe with YES/NO status
            results_data = []
            for i, class_name in enumerate(st.session_state.model_class_names):
                score = st.session_state.all_scores[class_name]
                threshold = st.session_state.model_thresholds[i]

                # Show "YES" if exceeded, "NO" if not
                status = "YES" if score >= threshold else "NO"

                results_data.append({
                    "Disease": class_name,
                    "Model Score": f"{score:.4f}",
                    "Threshold": f"{threshold:.4f}",
                    "Status": status
                })

            df = pd.DataFrame(results_data)

            # Display table with highlighting for YES rows
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=500
            )

            # Add custom CSS for row highlighting
            st.markdown("""
            <script>
            const cells = document.querySelectorAll('td');
            cells.forEach(cell => {
                if (cell.textContent.trim() === 'YES') {
                    cell.closest('tr').style.backgroundColor = 'rgba(76, 175, 80, 0.2)';
                }
            });
            </script>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error displaying results: {e}")

    # ====================================================================
    # VISUALIZATION SECTION
    # ====================================================================
    if st.session_state.get("show_visualization", False):
        st.markdown("### Model Output Scores")
        st.write("Comparison of model scores vs decision thresholds for each disease class")

        try:
            fig, ax = plt.subplots(figsize=(12, 5))

            diseases = list(st.session_state.all_scores.keys())
            scores = list(st.session_state.all_scores.values())
            thresholds = st.session_state.model_thresholds

            x_pos = np.arange(len(diseases))

            # Create bars with color based on threshold exceeded
            colors = ['#4CAF50' if scores[i] >= thresholds[i] else '#9E9E9E' 
                     for i in range(len(scores))]

            ax.bar(x_pos, scores, label="Model Score", alpha=0.85, color=colors)
            ax.plot(x_pos, thresholds, 'r--', linewidth=2.5, marker='o', 
                   markersize=6, label="Decision Threshold")

            ax.set_xlabel("Disease Class", fontsize=11, fontweight='bold')
            ax.set_ylabel("Score Value", fontsize=11, fontweight='bold')
            ax.set_title("Disease Score vs Threshold Comparison", fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1.0)

            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Visualization error: {e}")

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if "show_detailed" not in st.session_state:
    st.session_state.show_detailed = False
if "show_visualization" not in st.session_state:
    st.session_state.show_visualization = False

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()