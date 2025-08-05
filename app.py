"""
Streamlit application for predicting toxicity.
Supports single image and batch inference with optional cross-validation.
"""

# --- Standard library imports ---
import glob
import os
import math
# --- Third-party imports ---
import streamlit as st
from PIL import Image
import pandas as pd
# --- Local application imports ---
from app.model import ImageClassifier, CVWrapper

# Configure Streamlit page layout and icon
st.set_page_config(
    page_icon="🦠",
    layout="centered",
)

# Sidebar settings: default directories for models and configs
checkpoint_dir = "models"
config_dir = "configs"

# Discover available model checkpoint files
model_paths = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

# Option to enable cross-validation mode
use_cv = st.sidebar.checkbox("Use cross-validation", value=True)

if use_cv:
    # Configure cross-validation parameters and wrapper
    cv_options = ["EfficientNetB0 CV", "ResNet18 CV"]
    cv_choice = st.sidebar.selectbox("CV model", cv_options)
    if cv_choice == "ResNet18 CV":
        checkpoint_dir = "models/cv_resnet18"
        config_file = "config_finetune_resnet18.yaml"
        n_folds = 2
    else:
        checkpoint_dir = "models/cv_efficientnet_b0"
        config_file = "config_best_model.yaml"
        n_folds = 2

    mode = st.sidebar.radio("CV inference mode", ["ensemble", "fold"], index=0)
    fold = None
    if mode == "fold":
        fold = st.sidebar.selectbox("Fold to use", list(range(1, n_folds + 1)))

    # Load cross-validation wrapper
    model = CVWrapper(
        config_path=os.path.join(config_dir, config_file),
        checkpoint_dir=checkpoint_dir,
        n_folds=n_folds,
        device="cpu"
    )
else:
    # Single-model selection: allow user to choose one checkpoint (Best Model default)
    display_names = ["Best Model"] + model_names
    selected_model = st.sidebar.selectbox(
        "Choose model",
        display_names,
        index=0,
        help="Select the checkpoint (.pth file), Best Model is default"
    )

    # Cache the model-loading function for performance
    @st.cache_resource
    def load_model(display_name: str) -> ImageClassifier:
        """
        Load an ImageClassifier based on the selected display name.
        """
        if display_name == "Best Model":
            model_path = os.path.join(checkpoint_dir, "efficientnet_b0_2025-07-31_a40e6c4_general_model.pth")
            config_path = os.path.join(config_dir, "config_best_model.yaml")
        else:
            model_name = display_name
            model_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
            arch = model_name.split("_")[0]
            config_path = os.path.join(config_dir, f"config_finetune_{arch}.yaml")
        return ImageClassifier(config_path=config_path, model_path=model_path, device="cpu")
    model = load_model(selected_model)



# Main application title
st.title("🧪 Toxicity Classifier")

# Define tabs for single-image and batch analysis
tab_single, tab_batch = st.tabs(["Single Image Analysis", "Batch Analysis"])

with tab_single:
    # Single Image Analysis UI
    st.markdown("Upload an image (.jpg, .png, .tif) to get the toxicity probability.")
    file = st.file_uploader(
        label="Image :",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    if file:
        img = Image.open(file).convert("RGB")
        with st.spinner("Running inference..."):
            if use_cv:
                if mode == "ensemble":
                    scores = model.predict(img, mode="ensemble")
                else:
                    scores = model.predict(img, mode="fold", fold=fold)
            else:
                scores = model.predict(img, return_dict=True)

        # Extract and remove vote from scores if present
        vote = scores.pop("vote", None)

        st.subheader("Results")
        for lbl, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{lbl.capitalize()}** : {prob:.1%}")

        if vote is not None:
            # Display the vote as a descriptive verdict
            verdict = "Toxic" if vote == 1 else "Non-toxic"
            st.markdown(f"Vote: **{verdict}**")

        st.image(img, caption="Uploaded image", use_container_width=True)

with tab_batch:
    # Batch Analysis UI
    st.markdown(
        """
        Upload multiple images of the same strain (≥5 images recommended).  
        The app displays predictions per image, then the average probability and a
        confidence interval (±1 standard deviation).
        """
    )

    with st.form(key="prediction_form", clear_on_submit=True):
        files = st.file_uploader(
            "Multiple images (.jpg, .png, .tif):",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            accept_multiple_files=True
        )
        submitted = st.form_submit_button("🔍 Analyze")

    if files and submitted:
        rows = []
        for f in files:
            img = Image.open(f).convert("RGB")
            if use_cv:
                scores = model.predict(img, mode=mode, fold=fold)
            else:
                scores = model.predict(img, return_dict=True)
            row = {"File": f.name}
            row.update(scores)
            rows.append(row)

        df = pd.DataFrame(rows)

        mean_tox = df["Toxic"].mean()
        std_tox  = df["Toxic"].std(ddof=0)
        ci       = std_tox

        st.subheader("Strain Summary")
        st.write(f"Average **toxic** probability: {mean_tox:.1%} ± {ci:.1%}")
        st.progress(mean_tox)

        if use_cv:
            # Use majority vote per image for CV mode
            votes = df["vote"].tolist()
            vote_global = int(sum(votes) >= math.ceil(len(votes) / 2))
            verdict = "Toxic" if vote_global else "Non-toxic"
        else:
            verdict = "Toxic" if mean_tox >= 0.49 else "Non-toxic"
        st.markdown(f"Verdict: **{verdict}**")

        st.dataframe(
            df.style.format({cls: "{:.1%}" for cls in model.classes})
        )


st.markdown(
    """
    <div style="
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        background-color: #E8F0FE;
        border: 1px solid #AECBFA;
    ">
      <span style="font-size:1.4rem;">🔬</span>
      <span style="font-size:1rem; color: #1967D2;">
        These models remain experimental and are not yet ready for direct use.
      </span>
    </div>
    <div style="
        margin-top:0.5rem;
        font-size:0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        background-color: #ecf0f6;
        border: 1px solid #adb0be;
    ">
      <a href="https://github.com/Tiits/microcoleus-project.git" target="_blank" style="color: #000000; text-decoration: none;">
        🔗 GitHub Repository
      </a>
    </div>
    """,
    unsafe_allow_html=True
)