# README.md

# AMR Prediction System

A machine learning system for predicting antimicrobial resistance from DNA sequences.

## Project Structure

```
amr-system/
├── data/                  # Data directory
│   ├── raw/              # Raw DNA sequence files
│   └── processed/        # Preprocessed datasets
├── models/               # Saved model files
├── notebooks/           
│   └── dna-predict-amr.ipynb  # Main analysis notebook
├── python-app/          # Web application
│   ├── app/
│   │   ├── __init__.py
│   │   └── routes.py
│   └── requirements.txt
└── README.md
```

## Workflow Steps

1. **Data Preparation**
   - Load DNA sequence data (FASTA format)
   - Extract AMR labels from metadata
   - Perform sequence quality control
   - Split into train/validation/test sets

2. **Feature Engineering**
   - Convert DNA sequences to numerical features
   - Extract k-mer frequencies
   - Generate sequence embeddings
   - Normalize features

3. **Model Development**
   - Train machine learning models
   - Evaluate model performance
   - Perform cross-validation
   - Fine-tune hyperparameters

4. **Prediction Pipeline**
   - Load trained model
   - Process new DNA sequences
   - Generate AMR predictions
   - Output resistance probabilities

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tmone/amr-system.git
   cd amr-system
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r python-app/requirements.txt
   ```

## Usage

1. **Using Jupyter Notebook**
   ```bash
   jupyter notebook notebooks/dna-predict-amr.ipynb
   ```

2. **Using Web Application**
   ```bash
   cd python-app
   python run.py
   ```

## Model Performance

- Accuracy: TBD
- Precision: TBD
- Recall: TBD
- F1 Score: TBD

## License

This project is licensed under the MIT License.