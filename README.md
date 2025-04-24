# Microcoleus Project

Projet de bachelor : identification de la cyanobactérie Microcoleus anatoxicus par deep learning.

---

## Structure du projet

- `data/raw/` : images brutes, telles que reçues (non versionnées)
- `data/processed/` : images prêtes pour l’entraînement (non versionnées)
- `notebooks/` : notebooks Jupyter pour l’exploration et le prototypage
- `src/` : scripts et modules Python
- `outputs/checkpoints/` : modèles sauvegardés (non versionnés)
- `outputs/figures/` : courbes, graphiques et résultats d’analyse
- `outputs/logs/` : journaux d’entraînement ou d’évaluation

---

## Installation

1. Créez un environnement virtuel et activez-le :
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
2. Installez les dépendances :
    ```
    pip install -r requirements.txt
    ```

---

## Utilisation des scripts

### Prétraitement des images

Pour prétraiter toutes les images du dossier `data/raw/` et sauvegarder le résultat dans `data/processed/` :
   ```
   python src/preprocessing.py
   ```

### Vérification et listing des images

Pour lister les images disponibles et vérifier l’intégrité des fichiers:
   ```
   python src/dataset.py
   ```

---

## Flux de travail rapide

1. Placer les données d’images brutes dans `data/raw/`
2. Vérifier et prétraiter les images à l’aide des scripts (`src/dataset.py`, `src/preprocessing.py`)
3. Explorer les données avec le notebook d’exploration (`notebooks/01_data_exploration.ipynb`)
4. Entraîner et évaluer les modèles à partir des scripts ou des notebooks dédiés
5. Documenter l’avancement et les problèmes rencontrés dans le logbook (`LOGBOOK.md`)

---

## Confidentialité

⚠️ Les données brutes ne doivent pas être versionnées sur GitHub.


