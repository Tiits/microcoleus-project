# Journal de bord – Projet Microcoleus

## Objectifs principaux

### 1. Constitution et préparation de la base de données d’images
- [x] Vérifier la qualité, l’annotation et l’homogénéité des données
- [x] Structurer l’arborescence de la base de données pour le projet

### 2. Exploration et prétraitement des données
- [x] Réaliser une exploration visuelle et statistique des images
- [x] Mettre en place et documenter une pipeline de prétraitement adapté

### 3. Développement et évaluation du modèle de classification
- [ ] Implémenter et comparer différents modèles de classification d’images (CNN, transfer learning)
- [ ] Définir les métriques d’évaluation pertinentes
- [ ] Entraîner, valider et comparer les performances des modèles

### 4. Analyse des résultats et interprétabilité
- [ ] Analyser les erreurs principales et les limites du modèle
- [ ] Utiliser des méthodes d’interprétabilité pour valider les prédictions

### 5. Validation sur échantillons indépendants
- [ ] Tester le modèle sur de nouveaux jeux de données (cultures unialgales et échantillons environnementaux)
- [ ] Documenter la robustesse et la généralisabilité de la pipeline

---

## Plan d’action détaillé

### 1. Organisation initiale & premiers tests

- [x] Prise de contact avec le laboratoire, clarification du cadre du projet
- [x] Étude bibliographique sur la classification d’images de cyanobactéries/micro-organismes
- [x] Recherche et téléchargement de jeux de données similaires (ex : Kaggle)
- [x] Réalisation de premiers tests de modèles CNN sur ces jeux de données externes
- [x] Réalisation de premiers tests de modèles pré-entraînés sur ces jeux de données externes
- [x] Mise en place de la structure du dépôt Git et création du logbook
- [x] Rédaction du README, du .gitignore et du requirements.txt

### 2. Préparation avant réception des données réelles

- [x] Créer un notebook d’exploration prêt à l’emploi (`notebooks/01_data_exploration.ipynb`) : chargement, affichage, stats de base
- [x] Écrire un script utilitaire pour vérifier et lister les images (`src/dataset.py`)
- [x] Préparer un squelette de pipeline de prétraitement d’images (`src/preprocessing.py`)
- [x] Définir et créer la structure des dossiers : `data/raw/`, `data/processed/`, `outputs/checkpoints/`, `outputs/configs/`, `splits/`
- [x] Compléter/mettre à jour le README et le logbook (explications, instructions)
- [x] Préparer un module/fonctions de calcul de métriques d’évaluation (`src/metrics.py`)
- [x] Créer un module de loading depuis splits (`src/data_loader.py`)
- [x] Implémenter le script de génération de splits (`src/split_data.py`)
- [x] Définir et ajouter le loader de config (`src/config_utils.py`)

### 3. Mise en place du workflow reproductible (après réception des données NZ)

- [x] Adapter les notebooks (`01_data_exploration.ipynb`, `02_train_baseline.ipynb`) pour importer la config et les modules
- [x] Utiliser `load_from_splits` pour charger les datasets train/val
- [x] Paramétrer ModelCheckpoint, sauvegarde config, figures et logs via `run_id`
- [x] Mettre à jour le guide de versionnage Git et appliquer le tagging avant chaque essai

### 4. À compléter dès réception de toutes les données réelles

- [x] Analyser la structure, la diversité et la qualité des images reçues
- [x] Lancer l’exploration statistique et visuelle du dataset
- [ ] Adapter les notebooks/scripts à la structure et aux formats réels

---

## Tâches et avancement

| Date       | Objectif                          | Tâche                                                                                                                                                                                                                                                                         | Statut | Problème/Résultat/Remarque                                      |
|------------|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-----------------------------------------------------------------|
| 2025-03-06 | Prise de contact                  | Prise d’information sur le sujet, échanges par mail                                                                                                                                                                                                                           | Fait   | Précisions obtenues sur les données et le cadre du projet       |
| 2025-03-06 | Revue préliminaire                | Recherche d’exemples de jeux de données similaires                                                                                                                                                                                                                            | Fait   | Tests initiaux sur des datasets “micro-organismes” de Kaggle    |
| 2025-03-07 | Prototypage                       | Implémentation d’un premier CNN basique                                                                                                                                                                                                                                       | Fait   | Résultats initiaux mitigés, visualisation des performances      |
| 2025-03-12 | Prise en main des données         | Réception d’exemples d’images du laboratoire                                                                                                                                                                                                                                  | Fait   | Stockage et visualisation des images test                       |
| 2025-03-18 | Revue de littérature scientifique | Étudier le contexte biologique du projet et la problématique des cyanobactéries toxiques                                                                                                                                                                                      | Fait   | Synthétise des méthodes existantes d’identification             |
| 2025-03-21 | Suivi collaboratif                | Relance et échanges pour suivi des images externes                                                                                                                                                                                                                            | Fait   | Attente de réception d’images de Nouvelle-Zélande et USA        |
| 2025-04-24 | Organisation du projet            | Mise en place de la structure du dossier Git                                                                                                                                                                                                                                  | Fait   | Initialisation du dépôt, ajout de .gitignore, README, venv, etc. |
| 2025-04-24 | Logbook & planification           | Définition des objectifs et création du logbook                                                                                                                                                                                                                               | Fait   | Liste d’objectifs structurée, template logbook validé           |
| 2025-04-24 | Exploration des données           | Création du notebook d’exploration (`notebooks/01_data_exploration.ipynb`)                                                                                                                                                                                                    | Fait   | Fichier créé                                                    |
| 2025-04-24 | Prétraitement des données         | Ajout du script utilitaire de listing et vérification d’images (`src/dataset.py`)                                                                                                                                                                                             | Fait   | Fichier créé                                                    |
| 2025-04-24 | Prétraitement des données         | Préparation du pipeline de prétraitement (`src/preprocessing.py`)                                                                                                                                                                                                             | Fait   | Fichier créé                                                    |
| 2025-04-24 | Organisation du projet            | Création de la structure des dossiers (`data/`, `outputs/`) avec fichiers `.gitkeep`                                                                                                                                                                                          | Fait   | Dossiers créés avec `.gitkeep`                                  |
| 2025-04-24 | Documentation                     | Mise à jour du README et du logbook (sections scripts, dossiers, workflow)                                                                                                                                                                                                    | Fait   | README enrichi, logbook mis à jour                              |
| 2025-04-24 | Évaluation modèle                 | Ajout du module de métriques (`src/metrics.py`)                                                                                                                                                                                                                               | Fait   | Fichier créé                                                    |
| 2025-04-24 | Prototypage de modèle             | Création du notebook d’entraînement baseline (`notebooks/02_train_baseline.ipynb`)                                                                                                                                                                                            | Fait   | Fichier créé                                                    |
| 2025-04-29 | Données de NZ Reçues              | Contient un total de 139 images (non_toxic: 66, toxic: 73)                                                                                                                                                                                                                    | Fait   | Modules créés                                                   |
| 2025-04-29 | Restructuration des dossiers      | Ajout de dossiers pour les données de chaque pays et de dossiers 'all'                                                                                                                                                                                                        | Fait   | Dossiers créés                                                  |
| 2025-04-29 | Adaptation du code                | Adaptation du code au données et à la nouvelle structure et aux données réels reçues                                                                                                                                                                                          | Fait   | Fichiers modifié                                                |
| 2025-04-29 | Premiers tests réels              | Premiers tests utilisants des données (NZ) éffectutés, résultat mitigé: accuracy:0.69, val_accuracy: 0.48                                                                                                                                                                     | Fait   | Résultats initiaux mitigés, trop peu d'image                    |
| 2025-04-29 | Pipeline reproductible            | Implémentation `src/config_utils.py`, `src/split_data.py`, `src/data_loader.py`                                                                                                                                                                                               | Fait   | Modules créés                                                   |
| 2025-04-29 | Intégration workflows             | Adaptation notebooks avec config, splits et run_id                                                                                                                                                                                                                            | Fait   | Tests NZ images OK (pipeline fonctionnelle)                     |
| 2025-04-30 | Acquisition de données         | Prise d’images au microscope (souches : 1, 6, 18, 27)                                                                                                                                                                                                                         | Fait     | Première session de collecte                                    |
| 2025-05-07 | Acquisition de données         | Prise d’images au microscope (souches : 5, 6_23-01-25, 5_DV_25-10-24, Schaf_from_28-16_1, Schaf_from_28-10_2)                                                                                                                                                                 | Fait     | Deuxième session de collecte                                    |
| 2025-05-08 | Acquisition de données         | Prise d’images au microscope (souches : 5_Axenic_1_well-B7_27-02-25, 7_Axenic_1_well-B7_27-02-25_CI, CVA96_50_MLA_03-04-25, Q1_well_5_14-03-25_CI, Q1_well_5_14-03-25)                                                                                                        | Fait     | Troisième session de collecte                                   |
| 2025-05-27 | Prétraitement des données | Ajout du script d’extraction `.lif` → `.tif` (`src/io/lif_extractor.py`)                                                                                                                                                                                                      | Fait   | Extraction automatisée mise en place                            |
| 2025-05-27 | Organisation du projet     | Création des dossiers `data/extracted/` et `data/unextracted/`                                                                                                                                                                                                                | Fait   | Ajout de `.gitkeep` pour garder les dossiers vides dans Git     |
| 2025-07-04 | Prétraitement des données      | Ajout du script Fiji (`src/io/fiji_extract_lif.py`)                                                                                                                                                                                                                           | Fait     | Extraction `.lif`→`.tif` automatisée via Fiji                      |
| 2025-07-04 | Prétraitement des données      | Extraction d’images via script Python et Fiji pour souches (1, 6, 18, 27, 5, 6_23-01-25, 5_DV_25-10-24, Schaf_from_28-16_1, Schaf_from_28-10_2, 5_Axenic_1_well-B7_27-02-25, 7_Axenic_1_well-B7_27-02-25_CI, CVA96_50_MLA_03-04-25, Q1_well_5_14-03-25_CI, Q1_well_5_14-03-25) | Fait     | Images stockées dans `data/extracted/python` et `data/extracted/fiji` |
| 2025-07-04 | Exploration des données         | Relance du notebook d’exploration avec les nouvelles données                                                                                                                                                                                                                  | Fait     | 249 images non_toxic & 342 images toxic générées                  |
| 2025-07-04 | Développement de modèle         | Exécution du notebook baseline (`02_train_baseline.ipynb`)                                                                                                                                                                                                                    | Fait     | Validation accuracy max ≈ 73.9 % (baseline, tag 2025-07-04_7c09ce4) |
| 2025-07-04 | Développement de modèle         | Exécution du notebook transfer learning (`03_transfer_learning_efficientnetb0.ipynb`)                                                                                                                                                                                         | Fait     | Validation accuracy stable à 63.6 % (transfer, tag 2025-07-04_7c09ce4) |
| 2025-07-06 | Prétraitement des données      | Création de `preprocessing_finetune.py` : pipeline avancée (stratification, aug. fortes)                                                                                                                                                                                      | Fait | Rotation 30°, shift 0.2, shear 0.2, zoom 0.2, flip H, brightness ±20 % |
| 2025-07-06 | Prétraitement des données      | Génération de splits stratifiés via `split_data_finetune.py`                                                                                                                                                                                                                  | Fait | Fichier de splits dédié pour Fine-Tune (train/val équilibré)          |
| 2025-07-06 | Développement de modèle        | Ajout du notebook `04_finetune_resnet50.ipynb` (ResNet50 pré-entraîné, fine-tuning 40 epochs)                                                                                                                                                                                         | Fait | Overfitting détecté : val accuracy ≈ 0.58, val_recall_toxic instable   |
| 2025-07-07 | Développement de modèle        | Ajout du notebook `05_finetune_resnet18.ipynb` (inculant direcment une pipeline de prétraitement + split) (ResNet18 pré-entraîné, fine-tuning 40 epochs)                                                                                                                              | Fait | Val accuracy 96.6 % ; 2 FP / 2 FN ; fiabilité à confirmer |




---

## Problèmes rencontrés & solutions

### 2025-07-04 — Extraction automatisée des fichiers `.lif`
**Problème :**
Les solutions « pure code » (Bio-Formats Java, `pylifreader`, `tifffile`…) n'offrent pas toutes les options disponibles dans Fiji ou gèrent imparfaitement certains fichiers Leica (canaux tronqués, erreurs d’endiannes, dimension mismatch...). ImageJ ou Fiji posent également certains problèmes notamment l'application implicite de LUT (comme 'Fire LUT' ou 'Green') ce qui implique que les TIFF obtenus présentent donc des intensités parfois biaisées (intensités non linéaires → biais d’entraînement) et une reproductibilité incertaine.

**Solution :** 
- **Branche Python** : script `src/io/lif_extractor.py` pour une extraction rapide sans LUT, avec contrôle visuel ponctuel.  
- **Branche Fiji** : script `src/io/fiji_extract_lif.py` qui pilote Bio-Formats dans Fiji (export 16-bit, LUT désactivée).  
Les images sont sauvegardées séparément (`data/extracted/python/` vs `data/extracted/fiji/`) afin de quantifier l’impact de chaque pipeline sur l’entraînement.

---

## Idées, questions et pistes à creuser

- (Idée ou question à explorer)
- (Autre remarque ou piste pour la suite)
