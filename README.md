# Early Detection of Breast Cancer using Hybrid CNN and Transformer Models on Mammograms

This repository contains the implementation of a Senior Design Project focused on early detection of breast cancer using hybrid CNN and Transformer models on mammogram images.

## Project Overview

Breast cancer is one of the leading causes of death among women worldwide. This project aims to develop and compare various deep learning models for the early detection of breast cancer from mammogram images. The models include:

1. Pure CNN model
2. Pure Transformer model
3. Hybrid Sequential CNN-Transformer model
4. Hybrid Parallel CNN-Transformer model

The project also incorporates Explainable AI techniques to provide insights into the model's decision-making process.

## Project Structure

SDP_Project_P3/
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Preprocessed datasets
├── models/                   # Saved model weights
├── results/                  # Results and visualizations
├── logs/                     # Log files
├── src/
│   ├── data/                 # Data processing scripts
│   │   ├── data_preprocessing.py
│   │   ├── data_loader.py
│   │   └── download_datasets.py
│   ├── models/               # Model implementations
│   │   ├── models.py
│   │   └── train.py
│   ├── utils/                # Utility functions
│   │   ├── evaluation_metrics.py
│   │   ├── explainable_ai.py
│   │   ├── statistical_comparison.py
│   │   └── generate_report.py
│   └── main.py               # Main training script
├── notebooks/                # Jupyter notebooks for exploration
├── run_project.py            # Project runner script
└── README.md                 # Project documentation


## Setup Instructions

### Prerequisites

- Python 3.9
- Git
- Pip (Python package installer)
- 8GB+ RAM recommended
- NVIDIA GPU with CUDA support (recommended for faster training)

### Windows Setup

1. **Clone the repository**

   Open Command Prompt and run: git clone https://github.com/r3ckl3ssr3v/breast-cancer-detection.git


2. **Create a virtual environment**

   Run: python -m venv venv


3. **Activate the virtual environment**

   Run: venv\Scripts\activate


4. **Install dependencies**

   Run: pip install -r requirements.txt

    If you have a CUDA-compatible GPU:

    Run: pip install torch torchvision torchaudio --extra-index-url

    Run: pip install torch torchvision torchaudio --extra-index-url URL_ADDRESS.pytorch.org/whl/cu113

    If you don't have a CUDA-compatible GPU:

    Run: pip install torch torchvision torchaudio

5. **Download datasets**

   Run: python src/data/download_datasets.py --datasets all

Note: Some datasets require manual download due to registration requirements.

6. **Preprocess data**

   Run: python src/data/data_preprocessing.py --dataset MIAS


### macOS/Linux Setup

1. **Clone the repository**

Open Terminal and run:

git clone

git clone https://github.com/r3ckl3ssr3v/breast-cancer-detection.git

2. **Create a virtual environment**

Run: python3 -m venv venv source venv/bin/activate

3. **Install dependencies**

Run: pip install -r requirements.txt

For macOS with M1/M2 chip: pip install torch torchvision torchaudio


4. **Create required directories**
Run: mkdir data data/raw data/processed models results logs src src/data src/models src/utils notebooks

5. **Download datasets**

Run: python src/data/download_datasets.py --datasets all

Note: Some datasets require manual download due to registration requirements.

6. **Preprocess data**
Run: python src/data/data_preprocessing.py --dataset MIAS


## Usage

### Training Models

To train all models:
# Windows
python src\main.py --model all --batch_size 32 --epochs 50 --explain

# macOS/Linux
python src/main.py --model all --batch_size 32 --epochs 50 --explain


To train a specific model:
# Windows
python src\main.py --model hybrid_sequential --batch_size 32 --epochs 50 --explain

# macOS/Linux
python src/main.py --model hybrid_sequential --batch_size 32 --epochs 50 --explain


### Evaluation and Analysis

To perform statistical comparison of models:
# Windows
python src\utils\statistical_comparison.py

# macOS/Linux
python src/utils/statistical_comparison.py


To generate a comprehensive report:
# Windows
python src\utils\generate_report.py

# macOS/Linux
python src/utils/generate_report.py


### Running the Complete Project

To run the entire project pipeline:
# Windows
python run_project.py
# macOS/Linux
python run_project.py


This will:
1. Download and preprocess the datasets
2. Train and evaluate all models
3. Perform statistical comparison
4. Generate a comprehensive report

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   - Ensure you have the correct CUDA version installed for your GPU
   - Try running with CPU only by adding `--device cpu` to commands

2. **Memory Errors**
   - Reduce batch size: `--batch_size 16` or lower
   - Use a smaller dataset: `--dataset mini-ddsm`

3. **Missing Libraries**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Some libraries may need to be installed separately depending on your system

4. **Dataset Download Issues**
   - Some datasets require manual download due to registration requirements
   - Check the console output for instructions on manual downloads

### Getting Help

If you encounter any issues not covered here, please:
1. Check the project documentation
2. Open an issue on the GitHub repository
3. Contact the project maintainers

## Results

The project evaluates models using multiple metrics:
- Accuracy
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1 Score (macro and weighted)
- ROC AUC
- Confusion Matrix

Explainable AI techniques are used to visualize the regions of interest in mammogram images that contribute to the model's decision.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Cancer Imaging Archive (TCIA) for providing the CBIS-DDSM dataset
- The Mammographic Image Analysis Society for the MIAS dataset
- The INbreast dataset contributors