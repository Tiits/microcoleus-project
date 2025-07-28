# Microcoleus Project

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x+](https://img.shields.io/badge/TensorFlow-2.19+-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/) 
[![PyTorch 2.x+](https://img.shields.io/badge/PyTorch-2.7+-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/install) 

A deep learning approach for identifying the cyanobacterium _Microcoleus anatoxicus_.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Workflow](#workflow)
- [Streamlit Web Application](#streamlit-web-application)
- [Known Limitations](#known-limitations)
- [Data Privacy](#data-privacy)

## Overview

This bachelor project aims to implement deep learning techniques to identify _Microcoleus anatoxicus_, a cyanobacterium species. The project uses computer vision and machine learning to assist in the automated identification process, making it easier for researchers to analyze samples.

### Key Features
- Automated extraction of TIFF images from Leica `.lif` files
- Image preprocessing pipeline optimized for microscopy data
- Multiple deep learning models for comparison (ResNet18, ResNet50, EfficientNetB0)
- Comprehensive data splitting and validation procedures
- Detailed experiment tracking and reproducibility measures

## Project Structure

```
microcoleus-project/
├── configs/           # YAML configuration files for experiments
├── data/
│   ├── raw/          # Original images (not versioned)
│   ├── processed/    # Preprocessed training data (not versioned)
│   ├── extracted/    # Extracted TIFF files (not versioned)
│   └── unextracted/  # Original .lif files (not versioned)
├── notebooks/        # Jupyter notebooks for analysis
├── src/             # Python source code
│   └── io/          # Input/output utilities
├── outputs/         # Experiment outputs (not versioned)
└── splits/          # Train/validation/test splits
```

## Getting Started

### Prerequisites
- Python 3.12 or higher
- Fiji/ImageJ (for RGB Hyperstack extraction)
- Git
- Virtual environment manager

### Quick Start
1. Clone the repository:
```bash
git clone https://github.com/Tiits/microcoleus-project.git
cd microcoleus-project
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow

### 1. Data Preparation
1. **Image Extraction**
   - For standard images:
     ```bash
     python src/io/lif_extractor.py --input your_file.lif --output data/extracted/
     ```
   - For RGB Hyperstack: Use Fiji with the provided script (see detailed instructions below)

2. **Data Organization**
   - Place extracted images in `data/raw/[country]` or `data/raw/all`
   - Ensure consistent naming conventions

### 2. Analysis Pipeline

`Data Splits` and `Preprocess Images` are only needed for notebooks `02` and `03`.

1. **Generate Data Splits**
   ```bash
   python src/split_data.py
   ```

2. **Preprocess Images**
   ```bash
   python src/preprocessing.py
   ```

3. **Explore Your Data**
   - Open `notebooks/01_data_exploration.ipynb`
   - Make sure to use the correct data path
   - Review data distribution and quality

4. **Train Models**
   - Start with `notebooks/02_train_baseline.ipynb`
   - For better performance: `notebooks/05_finetune_resnet18.ipynb`
   - Always make sure to use the correct data path in the `config` file

5. **Evaluate Results**
   - Check `outputs/figures/` for visualizations
   - Review metrics in `outputs/logs/`

### RGB Hyperstack Extraction with Fiji
1. Download Fiji from [https://fiji.sc](https://fiji.sc)
2. Install the provided script:
   - Copy `fiji_extract_lif.py` to Fiji.app/scripts/
   - Or open via File ▸ New ▸ Script...
3. Run: Plugins ▸ Scripts ▸ fiji_extract_lif.py

## Streamlit Web Application

### Online Access
The application is available online at: `[URL]`

### Local Installation and Usage

1. **Install Streamlit**
```bash
pip install streamlit
```

2. **Run the App Locally**
```bash
cd microcoleus-project
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run app.py
```
The app will be available at `http://localhost:8501`

### Features
- Upload and analyze microscopy images
- Real-time predictions using pre-trained models
- Visualization of model confidence scores
- Batch processing capabilities
- Export results

### Adding Your Own Models

1. **Prepare Your Model**
- Export your trained model in one of the supported formats:
  ```python
  # For TensorFlow models
  model.save('path/to/your/model')
  
  # For PyTorch models
  torch.save(model.state_dict(), 'path/to/your/model.pth')
  ```

2. **Add Model to the App**
- Place your model in the `models/` directory
- Update the model configuration in `configs/`:

3. **Custom Preprocessing (Optional)**
If your model requires custom preprocessing:
```python
# In app/model.py
def your_custom_preprocessing(image):
    # Your preprocessing steps
    return processed_image
# Add the function in the Classifier class
```

### Troubleshooting
Common issues and solutions:
1. **Memory Issues**
   ```bash
   streamlit run app.py --server.maxUploadSize=1024
   ```

2. **GPU Support**
   - Ensure CUDA is properly installed
   - Set `use_gpu: true` in config.yaml

3. **Model Loading Errors**
   - Check model format compatibility
   - Verify dependencies versions

## Known Limitations

⚠️ **Important Notes for Users**

1. **Overfitting Concerns**
   - Current models show signs of overfitting
   - Recommended mitigations:
     - Use strong regularization
     - Implement early stopping
     - Reduce model complexity
     - Increase data augmentation

2. **Best Practices**
   - Always validate results manually
   - Use cross-validation
   - Monitor training curves carefully
   - Document any anomalies

## Data Privacy

⚠️ **Important:**
- Raw data files should never be committed to GitHub
- Trained models should be stored locally only
- Use `.gitignore` to prevent accidental uploads

## Support

For technical issues:
1. Check existing documentation in `notebooks/`
2. Review `LOGBOOK.md` for known issues
3. Open an issue on GitHub

---

*This project is part of a bachelor thesis in data science and machine learning.*