# app.py
import streamlit as st
import glob
import os
from PIL import Image
import pandas as pd
from app.model import Classifier, CVWrapper

# Configuration de la page Streamlit
st.set_page_config(
    page_icon="🦠",
    layout="centered",
)

st.sidebar.title("⚙️ Configuration")
checkpoint_dir = "models"
config_dir = "configs"

# Liste tous les .pth dans /models
model_paths = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

# Option Cross-validation
use_cv = st.sidebar.checkbox("Utiliser le ResNet18 CV", value=False)

if use_cv:
    # On propose le dossier contenant les ckpts fold1_model.pth...
    checkpoint_dir = st.sidebar.selectbox("Dossier CV", ["models/cv_resnet18"])
    n_folds = 2
    mode = st.sidebar.radio("Mode inférence CV", ["ensemble", "fold"], index=0)
    fold = None
    if mode == "fold":
        fold = st.sidebar.selectbox("Fold à utiliser", list(range(1, n_folds + 1)))
    # Charger le wrapper CV
    model = CVWrapper(
        config_path=os.path.join(config_dir, "config_finetune_resnet18.yaml"),
        checkpoint_dir=checkpoint_dir,
        n_folds=n_folds,
        device="cpu"
    )
else:

    # Sélecteur de checkpoint
    selected_model = st.sidebar.selectbox(
        "Choisir le modèle à utiliser",
        model_names,
        help="Sélectionnez le checkpoint (.pth)"
    )

    @st.cache_resource
    def load_model(model_name: str) -> Classifier:
        """
        Charge un Classifier en fonction du nom du modèle sélectionné.
        Extrait l'architecture (resnet18 ou resnet50) pour déterminer
        le fichier de config approprié.
        """
        model_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
        arch = model_name.split("_")[0]  # ex. "resnet18" ou "resnet50"
        config_path = os.path.join(config_dir, f"config_finetune_{arch}.yaml")
        return Classifier(config_path=config_path, model_path=model_path, device="cpu")
    # Chargement du modèle
    model = load_model(selected_model)


st.title("🦠 Microcoleus anatoxicus Toxicity Classifier")

# Deux onglets : image unique vs batch
tab_single, tab_batch = st.tabs(["Analyse d'une image", "Analyse batch"])

with tab_single:
    st.markdown("Chargez une image (.jpg, .png, .tif) pour obtenir la probabilité de toxicité.")
    file = st.file_uploader(
        label="Image :",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    if file:
        img = Image.open(file).convert("RGB")
        with st.spinner("Inférence en cours…"):
            if use_cv:
                if mode == "ensemble":
                    scores = model.predict(img, mode="ensemble")
                else:
                    scores = model.predict(img, mode="fold", fold=fold)
            else:
                scores = model.predict(img, return_dict=True)


        st.subheader("Résultats")
        for lbl, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{lbl.capitalize()}** : {prob:.1%}")

        st.image(img, caption="Image analysée", use_container_width=True)

# --- 2. BATCH MODE ------------------------------------------------------------
with tab_batch:
    st.markdown(
        """
        Chargez plusieurs champs de la **même souche** (≥5 images recommandées).  
        L’application affiche la prédiction par image puis la probabilité moyenne et un
        intervalle de confiance (±1 écart‑type).
        """
    )

    with st.form(key="prediction_form", clear_on_submit=True):
        files = st.file_uploader(
            "Images multiples (.jpg, .png, .tif) :",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            accept_multiple_files=True
        )
        submitted = st.form_submit_button("🔍 Analyser")

    if files and submitted:
        rows = []
        for f in files:
            img = Image.open(f).convert("RGB")
            if use_cv:
                scores = model.predict(img, mode=mode, fold=fold)
            else:
                scores = model.predict(img, return_dict=True)
            # crée une ligne avec le nom de fichier + toutes les classes
            row = {"Fichier": f.name}
            row.update(scores)
            rows.append(row)

        df = pd.DataFrame(rows)

        # ----- Agrégation -----
        # on suppose que la classe "Toxique" existe dans model.classes
        mean_tox = df["Toxique"].mean()
        std_tox  = df["Toxique"].std(ddof=0)
        ci       = std_tox  # ±1σ ~68% CI

        st.subheader("Synthèse souche")
        st.write(f"Probabilité moyenne **toxique** : {mean_tox:.1%} ± {ci:.1%}")
        st.progress(mean_tox)

        if use_cv:
            vote_global = int(mean_tox >= 0.49)
            verdict = "Toxique" if vote_global else "Non toxique"
        else:
            verdict = "Toxique" if mean_tox >= 0.49 else "Non toxique"
        st.markdown(f"Verdict : **{verdict}**")

        # affichage du tableau détaillé
        st.dataframe(
            df.style.format({cls: "{:.1%}" for cls in model.classes})
        )
