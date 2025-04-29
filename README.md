# Microcoleus Project

Projet de bachelor: identification de la cyanobactérie Microcoleus anatoxicus par deep learning.

---

## Structure du projet

- `configs/` : fichiers de configuration YAML pour chaque expérience
- `splits/` : définitions des découpages train/val/test (fixés pour la reproductibilité)
- `data/raw/` : images brutes, telles que reçues (non versionnées)
- `data/processed/` : images prétraitées prêtes pour l’entraînement (non versionnées)
- `notebooks/` : notebooks Jupyter pour l’exploration et le prototypage
- `src/`: scripts et modules Python (prétraitement, chargement, entraînement, etc.)
- `outputs/checkpoints/` : modèles sauvegardés par run (non versionnés)
- `outputs/configs/` : copies des fichiers de configuration utilisés pour chaque expérience
- `outputs/figures/` : graphiques, courbes et visualisations
- `outputs/logs/` : journaux d’entraînement, rapports et métriques

---

## Installation

1. Cloner le dépôt et se placer à la racine :
   ```bash
   git clone https://github.com/Tiits/microcoleus-project.git
   cd microcoleus-project
   ```
2. Créer un environnement virtuel et l’activer :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation des scripts pour expérimentation

### 1. Valider les changements et la configuration

Pour s'assurer que la version actuelle est committée :
   ```bash
  git status
  git add .
  git commit -m "chore: prepare for experiment – update config/model code"
  ```

### 2. Choisir un identifiant d’expérience
Récupérer le dernier run_id depuis un notebook:
  ```bash
  run_id = "yyyy-mm-dd_hash-git"
  ```

### 3. Taguer cette version
Pour créer un tag Git qui référence précisément ce commit et cette expérience :
  ```bash
  git tag -a exp/baseline/${run_id} -m "exp/baseline ${run_id}"
  git push origin main --tags
  ```

### 4. Listing et vérification

Pour lister les images disponibles et vérifier leur intégrité (si nécessaire) :
  ```bash
  python src/dataset.py
  ```

### 5. Prétraitement des images

Pour prétraiter toutes les images de `data/raw/all` vers `data/processed/all` (si nécessaire) :
  ```bash
  python src/preprocessing.py
  ```

### 6. Génération des splits

Pour créer les fichiers `splits/train.txt`, `splits/val.txt` et `splits/test.txt` (si nécessaire) :
  ```bash
  python src/split_data.py
  ```

### 7. Exploration des données

Ouvrir le notebook :
  ```
  notebooks/01_data_exploration.ipynb
  ```

### 8. Entraînement baseline

Lancer le notebook :
  ```
  notebooks/02_train_baseline.ipynb
  ```

### 9. Pour chaque nouvel essai
En cas de modifications de la config :
  ```bash
  git add configs/my_experiment.yaml
  git commit -m "feat: add ${config_name} config for experiment X"
  ```

---

## Workflow

1. Placer les données brutes dans `data/raw/{pays}` ou `data/raw/all`
2. Générer les splits pour train/val/test (`src/split_data.py`)
3. Prétraiter (`src/preprocessing.py`)
4. Explorer (`notebooks/01_data_exploration.ipynb`)
5. Entraîner et évaluer (`notebooks/02_train_baseline.ipynb`, autres notebooks)
6. Examiner les artefacts générés dans `outputs/`
7. Documenter l’avancement et les problèmes rencontrés dans `LOGBOOK.md`

---

## Confidentialité

⚠️ Les données brutes et les modèles entraînés **ne doivent pas** être versionnées sur GitHub.
