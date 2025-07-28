# Logbook – Microcoleus Project
## Main Objectives
### 1. Building and Preparing the Image Database
- [x] Verify data quality, annotations, and consistency
- [x] Structure the database hierarchy for the project
### 2. Data Exploration and Preprocessing
- [x] Perform visual and statistical data exploration
- [x] Implement and document an appropriate preprocessing pipeline
### 3. Model Development and Evaluation
- [x] Implement and compare various image classification models (CNN, transfer learning)
- [x] Define relevant evaluation metrics
- [x] Train, validate, and compare model performances
### 4. Results Analysis and Interpretability
- [x] Analyze main errors and model limitations
- [x] Use interpretability methods to validate predictions
### 5. Validation on Independent Samples
- [x] Test the model on new datasets
- [ ] Document pipeline robustness and generalizability

---
## Detailed Action Plan
### 1. Initial Setup & Preliminary Tests
- [x] Contact the lab and clarify project scope
- [x] Conduct literature review on image classification of cyanobacteria/microorganisms
- [x] Search and download similar datasets (e.g., Kaggle)
- [x] Run initial CNN model tests on external datasets
- [x] Run initial pretrained model tests on external datasets
- [x] Set up Git repository structure and create the logbook
- [x] Write README, .gitignore, and requirements.txt
### 2. Preparation Before Receiving Real Data
- [x] Create an exploratory notebook (`notebooks/01_data_exploration.ipynb`): loading, displaying, basic stats
- [x] Write a utility script to verify and list images (`src/dataset.py`)
- [x] Prepare a skeleton image preprocessing pipeline (`src/preprocessing.py`)
- [x] Define and create folder structure: `data/raw/`, `data/processed/`, `outputs/checkpoints/`, `outputs/configs/`, `splits/`
- [x] Update README and logbook (explanations and instructions)
- [x] Prepare an evaluation metrics module (`src/metrics.py`)
- [x] Create a data loading module from splits (`src/data_loader.py`)
- [x] Implement split generation script (`src/split_data.py`)
- [x] Add configuration loader (`src/config_utils.py`)
### 3. Establishing a Reproducible Workflow (After Receiving NZ Data)
- [x] Adapt notebooks (`01_data_exploration.ipynb`, `02_train_baseline.ipynb`) to import config and modules
- [x] Use `load_from_splits` to load training/validation datasets
- [x] Configure ModelCheckpoint, config saving, figures, and logs via `run_id`
- [x] Update Git versioning guide and apply tagging before each run
### 4. To Complete Upon Receiving All Real Data
- [x] Analyze structure, diversity, and quality of received images
- [x] Run statistical and visual data exploration
- [x] Adapt notebooks/scripts to actual data structures and formats
- [x] Enhance post-training metrics and visualizations
- [x] Implement callbacks to log MCC and balanced accuracy
- [x] Set up interpretability via Grad-CAM
- [x] Automate optimal decision threshold search via PR-curve
- [x] Integrate focal loss and optimize overfitting reduction
- [x] Implement stratified k-fold cross-validation
- [x] Experiment with spatial dropout (RandomErasing) to reduce overfitting
- [x] Automate optimal threshold search via PR-curve in `05_finetune_resnet18.ipynb`

---

## Tasks and Progress
| Date       | Objective                     | Task                                                                                                                                                                                                                                             | Status | Issue/Result/Note                                     |
|------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-------------------------------------------------------|
| 2025-03-06 | Initial Contact               | Gather project information, email exchanges                                                                                                                                                                                                     | Done   | Clarifications obtained on data and project scope     |
| 2025-03-06 | Preliminary Review            | Search for examples of similar datasets                                                                                                                                                                                                          | Done   | Initial tests on “micro-organisms” datasets from Kaggle |
| 2025-03-07 | Prototyping                   | Implementation of a basic CNN                                                                                                                                                                                                                    | Done   | Initial results mixed, performance visualization      |
| 2025-03-12 | Data Familiarization          | Receive sample images from the lab                                                                                                                                                                                                               | Done   | Storage and display of test images                    |
| 2025-03-18 | Literature Review             | Study biological context and toxic cyanobacteria identification methods                                                                                                                                                                          | Done   | Summary of existing identification methods            |
| 2025-03-21 | Collaborative Follow-up       | Follow-up and exchanges on external image sources                                                                                                                                                                                                | Done   | Awaiting image reception from New Zealand and USA     |
| 2025-04-24 | Project Organization          | Set up Git folder structure                                                                                                                                                                                                                      | Done   | Repository initialized, .gitignore, README, venv added |
| 2025-04-24 | Logbook & Planning            | Define objectives and create logbook                                                                                                                                                                                                              | Done   | Structured list of objectives, logbook template validated |
| 2025-04-24 | Data Exploration              | Create exploration notebook (`notebooks/01_data_exploration.ipynb`)                                                                                                                                                                               | Done   | File created                                          |
| 2025-04-24 | Data Preprocessing            | Add image listing and verification script (`src/dataset.py`)                                                                                                                                                                                     | Done   | File created                                          |
| 2025-04-24 | Data Preprocessing            | Prepare preprocessing pipeline (`src/preprocessing.py`)                                                                                                                                                                                          | Done   | File created                                          |
| 2025-04-24 | Project Setup                 | Create folder structure (`data/`, `outputs/`) with `.gitkeep`                                                                                                                                                                                     | Done   | Folders created with `.gitkeep`                       |
| 2025-04-24 | Documentation                 | Update README and logbook (scripts, folders, workflow sections)                                                                                                                                                                                  | Done   | README enhanced, logbook updated                      |
| 2025-04-24 | Model Evaluation              | Add metrics module (`src/metrics.py`)                                                                                                                                                                                                            | Done   | File created                                          |
| 2025-04-24 | Model Prototyping             | Create baseline training notebook (`notebooks/02_train_baseline.ipynb`)                                                                                                                                                                           | Done   | File created                                          |
| 2025-04-29 | NZ Data Received              | Contains total of 139 images (non_toxic: 66, toxic: 73)                                                                                                                                                                                           | Done   | Modules created                                       |
| 2025-04-29 | Folder Restructure            | Add country-specific and "all" folders                                                                                                                                                                                                            | Done   | Folders created                                       |
| 2025-04-29 | Code Adaptation               | Adapt code to new structure and real data                                                                                                                                                                                                         | Done   | Files modified                                        |
| 2025-04-29 | First Real Tests              | Initial tests using real NZ data: accuracy: 0.69, val_accuracy: 0.48                                                                                                                                                                               | Done   | Initial mixed results, too few images                 |
| 2025-04-29 | Reproducible Pipeline         | Implement `src/config_utils.py`, `src/split_data.py`, `src/data_loader.py`                                                                                                                                                                        | Done   | Modules created                                       |
| 2025-04-29 | Workflow Integration          | Adapt notebooks with config, splits, and run_id                                                                                                                                                                                                  | Done   | NZ image tests OK (pipeline functional)               |
| 2025-04-30 | Data Acquisition             | Microscopy imaging (strains: 1, 6, 18, 27)                                                                                                                                                                                                        | Done   | First collection session                              |
| 2025-05-07 | Data Acquisition             | Microscopy imaging (strains: 5, 6_23-01-25, 5_DV_25-10-24, Schaf_from_28-16_1, Schaf_from_28-10_2)                                                                                                                                                   | Done   | Second collection session                             |
| 2025-05-08 | Data Acquisition             | Microscopy imaging (strains: 5_Axenic_1_well-B7_27-02-25, 7_Axenic_1_well-B7_27-02-25_CI, CVA96_50_MLA_03-04-25, Q1_well_5_14-03-25_CI, Q1_well_5_14-03-25) | Done   | Third collection session                              |
| 2025-05-27 | Data Preprocessing            | Add `.lif`→`.tif` extraction script (`src/io/lif_extractor.py`)                                                                                                                                                                                   | Done   | Automated extraction in place                         |
| 2025-05-27 | Project Organization          | Create `data/extracted/` and `data/unextracted/` folders                                                                                                                                                                                          | Done   | Added `.gitkeep` to retain empty folders in Git        |
| 2025-07-04 | Data Preprocessing            | Add Fiji extraction script (`src/io/fiji_extract_lif.py`)                                                                                                                                                                                         | Done   | `.lif`→`.tif` extraction automated via Fiji            |
| 2025-07-04 | Data Preprocessing            | Extract images via Python and Fiji for strains (1, 6, 18, 27, 5, 6_23-01-25, 5_DV_25-10-24, Schaf_from_28-16_1, Schaf_from_28-10_2, 5_Axenic_1_well-B7_27-02-25, 7_Axenic_1_well-B7_27-02-25_CI, CVA96_50_MLA_03-04-25, Q1_well_5_14-03-25_CI, Q1_well_5_14-03-25) | Done | Images stored in `data/extracted/python` and `data/extracted/fiji` |
| 2025-07-04 | Data Exploration              | Re-run exploration notebook with new data                                                                                                                                                                                                        | Done   | 249 non_toxic images & 342 toxic images generated     |
| 2025-07-04 | Model Development             | Execute baseline notebook (`02_train_baseline.ipynb`)                                                                                                                                                                                             | Done   | Max validation accuracy ≈ 73.9% (baseline, tag 2025-07-04_7c09ce4) |
| 2025-07-04 | Model Development             | Execute transfer learning notebook (`03_transfer_learning_efficientnetb0.ipynb`)                                                                                                                                                                   | Done   | Stable validation accuracy at 63.6% (transfer, tag 2025-07-04_7c09ce4) |
| 2025-07-06 | Data Preprocessing            | Create `preprocessing_finetune.py`: advanced pipeline (stratification, strong augmentations)                                                                                                                                                      | Done   | Rotation 30°, shift 0.2, shear 0.2, zoom 0.2, horizontal flip, brightness ±20% |
| 2025-07-06 | Data Preprocessing            | Generate stratified splits via `split_data_finetune.py`                                                                                                                                                                                           | Done   | Dedicated splits file for fine-tuning (balanced train/val) |
| 2025-07-06 | Model Development             | Add notebook `04_finetune_resnet50.ipynb` (ResNet50 pretrained, fine-tuning 40 epochs)                                                                                                                                                            | Done   | Overfitting detected: val accuracy ≈ 0.58, unstable val_recall_toxic |
| 2025-07-06 | Class Balancing               | Implement dynamic oversampling of minority class in `train_df` + recalculate `weights`                                                                                                                                                            | Done   | Classes balanced, toxic recall increased               |
| 2025-07-06 | Data Pipeline `tf.data`       | Switch from `ImageDataGenerator` to `tf.data` pipeline (TIFF support, strong augmentations, prefetch)                                                                                                                                            | Done   | 25% faster reading, stable memory usage                 |
| 2025-07-06 | Advanced Metrics & Callback   | Update `src/metrics.py` (ROC/PR, MCC, calibration) + add `F1MCCCallback`                                                                                                                                                                            | Done   | `val_mcc` and `val_bal_acc` logged each epoch        |
| 2025-07-06 | Keras Warning Handling        | Suppress "PyDataset class should call super().__init__" warning via `weight_col`                                                                                                                                                                   | Done   | Warning gone, clean pipeline                           |
| 2025-07-06 | Focal Loss & Callbacks        | Adopt `BinaryFocalCrossentropy` + EarlyStopping/ModelCheckpoint on `val_pr_auc`                                                                                                                                                                   | Done   | Increased PR-AUC, reduced overfitting                   |
| 2025-07-07 | Model Development             | Add notebook `05_finetune_resnet18.ipynb` (includes pipeline, splits) (ResNet18 pretrained, fine-tuning 40 epochs)                                                                                                                                 | Done   | Val accuracy 96.6%; 2 FP / 2 FN; reliability to confirm |
| 2025-07-08 | Model Development             | Update `05_finetune_resnet18.ipynb` (add ROC AUC, PR curves, early stopping, calibration)                                                                                                                                                         | Done   | Enhanced validation with AUC/PR and overfitting control |
| 2025-07-08 | Model Development             | Experiment `data/raw/all` and `data/raw/Switzerland` with `05_finetune_resnet18.ipynb`                                                                                                                                                            | Done   | Similar performances                                    |
| 2025-07-10 | Grad-CAM & Interpretability   | Integrate `pytorch-grad-cam`, HD heatmaps in grid                                                                                                                                                                                                 | Done   | Clear visualization, more concise code                 |
| 2025-07-10 | Calibrated ResNet18 Evaluation | Early stopping + Platt calibration, new evaluation (Val acc 0.983, AUC/AP 1.00)                                                                                                                                                                     | Done   | Reliable model, optimal threshold retained             |
| 2025-07-10 | Reproducibility               | Document and enable deterministic options (seeds, `TF_DETERMINISTIC_OPS`)                                                                                                                                                                         | Done   | Results reproducible on GPU                             |
| 2025-07-18 | Data Acquisition             | Microscopy imaging (strains: 1, CYA4#40, CYA7#53, CYA8#56, CYA9#58, CYA10#59)                                                                                                                                                                      | Done   | Fourth collection session                              |
| 2025-07-21 | Cross-Validation ResNet18     | Add notebook `06_finetune_resnet18_cross-validation.ipynb`                                                                                                                                                                                         | Done   | Model generalization issues                             |
| 2025-07-21 | Streamlit CV App             | Create `app.py` and `app/` package for a CV app                                                                                                                                                                                                  | Done   | Demonstration app functional                           |
| 2025-07-24 | Cross-Validation ResNet18     | Experiment with `label_smoothing=0.1` and fold configuration                                                                                                                                                                                      | Done   | Improved model generalization                          |
| 2025-07-25 | Threshold Optimization        | Implement `best_threshold` in ResNet18 cross-validation                                                                                                                                                                                           | Done   | Optimized decision threshold                           |
| 2025-07-26 | Streamlit Improvement         | Update app with cross-validation and adjust `fc.1.weight`                                                                                                                                                                                         | Done   | Improved UI                                            |
| 2025-07-27 | Architecture Comparison       | Create notebook `07_architectures_comparison.ipynb` and YAML config                                                                                                                                                                               | Done   | Basis for model benchmarking                           |
| 2025-07-28 | Notebook Standardization      | Refactor, translate to English, unify structure/style, YAML docs                                                                                                                                                                                  | Done   | Professional, coherent documentation                    |

## Issues Encountered & Solutions
### 2025-07-04 — Automated `.lif` File Extraction
**Issue:**  
Pure code solutions (Bio-Formats Java, `pylifreader`, `tifffile`, etc.) do not offer all options available in Fiji or handle some Leica files imperfectly (truncated channels, endianness errors, dimension mismatches). ImageJ/Fiji also pose issues with implicit LUT application (e.g., 'Fire LUT', 'Green'), resulting in TIFFs with potentially biased intensities (non-linear intensities → training bias) and uncertain reproducibility.

**Solution:**  
- **Python branch:** `src/io/lif_extractor.py` for quick extraction without LUT, with occasional visual checks.  
- **Fiji branch:** `src/io/fiji_extract_lif.py` controlling Bio-Formats in Fiji (export 16-bit, LUT disabled).  
Images are saved separately (`data/extracted/python/` vs `data/extracted/fiji/`) to quantify each pipeline's impact on training.

### 2025-07-09 — Class Imbalance & Unstable Toxic Recall
**Issue:**  
Extreme oscillations in `val_recall_toxic` due to strong imbalance (toxic ≫ non-toxic).

**Solution:**  
- Oversample the minority class until balanced.  
- Recompute class weights.  
- Automatic threshold calculation optimizing F1 via PR-curve.

### 2025-07-10 — GPU/CuDNN Nondeterminism
**Issue:**  
Result variability between runs (GPU kernels non-deterministic, TensorFlow multithreading).

**Solution:**  
- Fix `random`, `numpy`, and `tensorflow` seeds.  
- Set `TF_DETERMINISTIC_OPS=1` and limit intra/inter-op threads.  
- Use stateless APIs and reduce `tf.data` pipeline parallelism.

---

## Ideas, Questions, and Future Directions
- (Idea or question to explore)  
- (Another remark or avenue for follow-up)  
