import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# 1. Device Setup (Apple Silicon MPS Acceleration)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚡ Using device: {device}")

# 2. Taxonomy Labels
BLOOMS_LABELS = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
label2id = {label: i for i, label in enumerate(BLOOMS_LABELS)}
id2label = {i: label for i, label in enumerate(BLOOMS_LABELS)}

# Load dataset from CSV
df = pd.read_csv("blooms_dataset.csv")
df["label"] = df["category"].map(label2id)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# 3. Tokenization
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def tokenize_function(examples):
    return tokenizer(examples["question"], truncation=True, max_length=128)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)

# 4. Model Initialization
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(BLOOMS_LABELS),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 5. Training Arguments (Increased epochs to 6 for higher accuracy)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=2,
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    print("\n🚀 Retraining DistilBERT on CSV dataset with M2 GPU...")
    trainer.train()

    SAVE_DIR = "./saved_blooms_model"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\n✅ Model fine-tuned on real dataset and saved to '{SAVE_DIR}'!")
