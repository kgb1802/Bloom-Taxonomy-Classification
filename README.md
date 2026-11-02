# 🎓 Bloom's Taxonomy Question Classification

An NLP-powered classification system that automatically categorizes academic questions into the six cognitive levels of **Bloom's Taxonomy** using a fine-tuned DistilBERT transformer model.

The project combines Natural Language Processing (NLP), Deep Learning, and an interactive Streamlit interface to help educators and academic professionals understand the cognitive complexity of assessment questions.

---

## 📌 Project Overview

Bloom's Taxonomy is a framework used to classify educational learning objectives according to their cognitive complexity.

This project uses a fine-tuned **DistilBERT sequence classification model** to analyze the semantic meaning of academic questions and predict their corresponding cognitive level.

The system supports:
- Real-time classification of individual questions.
- Batch classification through CSV file uploads.
- Confidence scores for all six cognitive categories.
- Exporting classified questions and predictions as CSV.

## 🧠 Bloom's Taxonomy Levels

The model classifies questions into the following six categories:

| Level | Cognitive Category | Description |
|---|---|---|
| 1 | Remember | Recall facts, concepts, and information. |
| 2 | Understand | Explain, interpret, and summarize concepts. |
| 3 | Apply | Use knowledge to solve problems or execute tasks. |
| 4 | Analyze | Examine relationships, patterns, and structures. |
| 5 | Evaluate | Critically assess, justify, and make decisions. |
| 6 | Create | Generate, design, or construct something new. |

## ⚙️ Technology Stack

**Machine Learning & NLP**
- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Hugging Face Datasets
- Scikit-learn

**Data Processing**
- Pandas
- NumPy

**Application Development**
- Streamlit

**Hardware Acceleration**
- Apple Silicon GPU (MPS), when available
- CPU fallback

## 🏗️ Model Architecture & Methodology

The project uses `distilbert-base-uncased` as its pretrained language model.

The model is fine-tuned for a six-class sequence classification task.

### Training Pipeline

1. **Dataset Loading:** Load academic questions and their corresponding Bloom's Taxonomy categories from `blooms_dataset.csv`.
2. **Label Encoding:** Convert the six cognitive categories into numerical labels.
3. **Dataset Splitting:** Split the dataset into 80% training and 20% validation subsets using stratified sampling.
4. **Tokenization:** Process questions using `DistilBertTokenizerFast`, with a maximum sequence length of 128 tokens.
5. **Model Fine-Tuning:** Train `DistilBertForSequenceClassification` using Hugging Face's Trainer API.
6. **Evaluation:** Evaluate the model using accuracy, precision, recall, and weighted F1-score.
7. **Model Saving:** Save the fine-tuned model and tokenizer for subsequent inference.

### Training Configuration

| Parameter | Value |
|---|---|
| Pretrained Model | distilbert-base-uncased |
| Learning Rate | 3e-5 |
| Training Epochs | 6 |
| Training Batch Size | 8 |
| Evaluation Batch Size | 8 |
| Maximum Token Length | 128 |
| Weight Decay | 0.01 |
| Validation Split | 20% |
| Random State | 42 |
| Model Selection Metric | Weighted F1-score |

## ✨ Application Features

### 1. Single Question Classification

Enter an academic question and receive:
- Predicted Bloom's Taxonomy level.
- Confidence score of the prediction.
- Probability distribution across all six cognitive categories.

### 2. Batch CSV Classification

Upload a CSV file containing a `question` column to classify multiple questions simultaneously.

The application generates:
- Predicted Bloom's Taxonomy category.
- Prediction confidence.
- Downloadable CSV containing the classification results.

## 📂 Project Structure

```text
Bloom-Taxonomy-Classification/
│
├── app.py                  # Streamlit application
├── train.py                # Model training pipeline
├── blooms_dataset.csv      # Training dataset
├── .gitignore              # Excluded files and directories
│
├── results/                # Training checkpoints (generated)
├── saved_blooms_model/     # Fine-tuned model (generated)
└── venv/                   # Python virtual environment
```

*Generated model weights, checkpoints, and virtual environments are excluded from Git version control.*

## 🚀 Getting Started

### Prerequisites

- Python
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/kgb1802/Bloom-Taxonomy-Classification.git

cd Bloom-Taxonomy-Classification
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install torch numpy pandas scikit-learn transformers datasets accelerate streamlit
```

### 4. Train the Model

```bash
python train.py
```

The training pipeline will fine-tune DistilBERT and save the model and tokenizer inside:

```text
saved_blooms_model/
```

### 5. Launch the Application

After successful training, run:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Evaluation Metrics

The training pipeline calculates the following evaluation metrics:

- **Accuracy:** Overall proportion of correctly classified questions.
- **Precision:** Proportion of correct positive predictions.
- **Recall:** Proportion of actual positive samples correctly identified.
- **Weighted F1-Score:** Weighted harmonic mean of precision and recall across all classes.

Actual evaluation scores should be recorded from the training run rather than assumed.

## 🔮 Future Enhancements

- Expand the training dataset with more diverse academic questions.
- Improve classification performance through hyperparameter optimization.
- Add detailed confusion matrix and class-wise performance visualization.
- Support multiple educational domains and question formats.
- Deploy the application as a publicly accessible web service.
- Explore explainable AI techniques to interpret classification decisions.

## 👨‍💻 Author

**KGB1802**

GitHub: [@kgb1802](https://github.com/kgb1802)

Project Repository: [Bloom-Taxonomy-Classification](https://github.com/kgb1802/Bloom-Taxonomy-Classification)

---

*Developed as an NLP and Deep Learning project exploring automated educational question classification using transformer-based language models.*
