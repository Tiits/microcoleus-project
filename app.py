import streamlit as st
import glob
import os
from PIL import Image
import pandas as pd
from app.model import Classifier

st.sidebar.title("⚙️ Configuration")
checkpoint_dir = "outputs/checkpoints/all/finetune_resnet18"

model_paths = glob.glob(os.path.join(checkpoint_dir, "*.pth"))

model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

selected_model = st.sidebar.selectbox(
    "Choisir le modèle à utiliser", model_names,
    help="Sélectionnez le checkpoint (.pth) pour le Classifier"
)

@st.cache_resource
def load_model(path: str) -> Classifier:
    return Classifier(
        model_path=path,
        config_path="configs/config_finetune_resnet18.yaml",
        device="cpu"
    )

model_path = os.path.join(checkpoint_dir, f"{selected_model}.pth")
model = load_model(model_path)

st.set_page_config(
    page_title="Microcoleus anatoxicus Toxicity Classifier",
    page_icon="🦠",
    layout="centered",
)

st.title("🦠 _Microcoleus anatoxicus_ Toxicity Classifier")

# ---------- UI LAYOUT ----------
tab_single, tab_batch = st.tabs(["Analyse d'une image", "Analyse d'une souche"])

# --- 1. SINGLE IMAGE ----------------------------------------------------------
with tab_single:
    st.markdown("Chargez une image microscopique pour obtenir la probabilité de toxicité.")

    file = st.file_uploader(
        "Image (.jpg, .png, .tif) :",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )

    if file:
        img = Image.open(file).convert("RGB")
        with st.spinner("Inférence en cours…"):
            scores = model.predict(img, return_dict=True)

        st.subheader("Résultats")
        for lbl, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{lbl.capitalize()}**: {prob:.1%}")

        st.image(img, caption="Image analysée", use_container_width=True)

# --- 2. BATCH MODE ------------------------------------------------------------
with tab_batch:
    st.markdown(
        """
        Chargez plusieurs champs de la **même souche** (≥5 images recommandées).  
        L’application affiche la prédiction par image puis la probabilité moyenne et un
        intervalle de confiance(±1 écart‑type).
        """
    )

    with st.form(clear_on_submit=True , key="prediction_form"):
        files = st.file_uploader(
            "Images multiples (.jpg, .png, .tif) :",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            accept_multiple_files=True
        )

        col1, col2 = st.columns([4, 1])

        with col2:
            submitted = st.form_submit_button("🔍 Analyser")


    if files and submitted:
        rows = []
        for f in files:
            img = Image.open(f).convert("RGB")
            scores = model.predict(img, return_dict=True)
            rows.append({"Fichier": f.name, "Toxique": scores["Toxic"], "Non toxique": scores["Non-toxic"]})

        df = pd.DataFrame(rows)

        # ----- Aggregation -----
        mean_toxic = df["Toxique"].mean()
        std_toxic = df["Toxique"].std(ddof=0)
        ci = 1.0 * std_toxic  # ±1σ  ~68% CI

        st.subheader("Synthèse souche")
        st.write(f"Probabilité moyenne **toxique**: {mean_toxic:.1%} ± {ci:.1%}")
        st.progress(mean_toxic)

        decision = "Toxique" if mean_toxic >= 0.5 else "Non toxique"
        st.markdown(f"Verdict: **{decision}**")

        st.dataframe(df.style.format({"Toxique": "{:.1%}", "Non toxique": "{:.1%}"}))


