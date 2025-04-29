# Microcoleus Project

Projet de bachelor : identification de la cyanobactérie Microcoleus anatoxicus par deep learning.

---

## Structure du projet

- `configs/` : fichiers de configuration YAML pour chaque expérience
- `splits/` : définitions des découpages train/val/test (fixés pour la reproductibilité)
- `data/raw/` : images brutes, telles que reçues (non versionnées)
- `data/processed/` : images prétraitées prêtes pour l’entraînement (non versionnées)
- `notebooks/` : notebooks Jupyter pour l’exploration et le prototypage
- `src/` : scripts et modules Python (prétraitement, chargement, entraînement, etc.)
- `outputs/checkpoints/all/` : modèles sauvegardés par run (non versionnés)
- `outputs/configs/all/` : copies des fichiers de configuration utilisés pour chaque expérience
- `outputs/figures/` : graphiques, courbes et visualisations
- `outputs/logs/` : journaux d’entraînement, rapports et métriques

---

## Installation

1. Cloner le dépôt et se placer à la racine :
   ```bash
   git clone <repo_url>
   cd microcoleus-project
   ```
2. Créer un environnement virtuel et l’activer :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation des scripts

### 1. Listing et vérification

Pour lister les images disponibles et vérifier leur intégrité :
  ```bash
  python src/dataset.py
  ```

### 2. Prétraitement des images

Pour prétraiter toutes les images de `data/raw/all` vers `data/processed/all` :
  ```bash
  python src/preprocessing.py
  ```

### 3. Génération des splits

Pour créer les fichiers `splits/train.txt`, `splits/val.txt` et `splits/test.txt` :
  ```bash
  python src/split_data.py
  ```

### 4. Exploration des données

Ouvrir le notebook :
  ```
  notebooks/01_data_exploration.ipynb
  ```

### 5. Entraînement baseline

Lancer le notebook :
  ```
  notebooks/02_train_baseline.ipynb
  ```

---

## Workflow

1. Placer les données brutes dans `data/raw/{pays}` ou `data/raw/all`
2. Générer les splits pour train/val/test (`src/split_data.py`)
3. Prétraiter (`src/preprocessing.py`)
4. Explorer (`notebooks/01_data_exploration.ipynb`)
5. Entraîner et évaluer (`notebooks/02_train_baseline.ipynb`, autres notebooks)
6. Examiner les artefacts générés dans `outputs/`
7. Documenter l’avancement et les problèmes rencontrés dans `LOGBOOK.md`

---

## Confidentialité

⚠️ Les données brutes et les modèles entraînés **ne doivent pas** être versionnées sur GitHub.
