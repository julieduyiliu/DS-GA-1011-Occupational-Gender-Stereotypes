#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# bert_finetune.py

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import pandas as pd
from config import Config
import utils

# Load and preprocess SST-2 dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["idx", "label", "dummy", "sentence"])
    df = df.drop(columns=["idx", "dummy"])  # Remove unnecessary columns
    return df

# Load train and validation data
train_df = load_data(Config.TSV_TRAIN)
dev_df = load_data(Config.TSV_DEV)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

# Apply tokenization
train_encodings = tokenizer(list(train_df["sentence"]), truncation=True, padding=True, max_length=Config.MAX_SEQUENCE_LENGTH)
dev_encodings = tokenizer(list(dev_df["sentence"]), truncation=True, padding=True, max_length=Config.MAX_SEQUENCE_LENGTH)

# Convert labels to tensor
train_labels = torch.tensor(train_df["label"].values)
dev_labels = torch.tensor(dev_df["label"].values)

# Create Dataset class for PyTorch
class SST2Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = SST2Dataset(train_encodings, train_labels)
dev_dataset = SST2Dataset(dev_encodings, dev_labels)

# Define accuracy metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Set up Trainer
training_args = TrainingArguments(
    output_dir="/content/bert_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()

# Evaluate on validation set and print accuracy
eval_results = trainer.evaluate()
print(f"Validation accuracy: {eval_results['eval_accuracy']:.4f}")

