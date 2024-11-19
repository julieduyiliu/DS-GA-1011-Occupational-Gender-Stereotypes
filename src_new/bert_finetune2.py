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

# Determine device and move model to it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

# Generate predictions for BERT_FILE
# First 800 predictions are for the validation set
dev_logits = trainer.predict(dev_dataset).predictions
dev_probs = torch.softmax(torch.tensor(dev_logits), dim=1)[:, 1].numpy()  # Probability of the positive class
preds_sst2 = list(dev_probs)[:800]  # Take only the first 800 as required

# Add control experiment predictions for "This man is a person." and "This woman is a person."
control_sentences = ["This man is a person.", "This woman is a person."] * 20  # Repeat 20 times each for a total of 40 sentences
control_encodings = tokenizer(control_sentences, truncation=True, padding=True, max_length=Config.MAX_SEQUENCE_LENGTH, return_tensors="pt")

# Move control_encodings to the same device as the model
control_encodings = {k: v.to(device) for k, v in control_encodings.items()}

# Generate predictions for control experiment
with torch.no_grad():
    control_outputs = model(**control_encodings)
control_probs = torch.softmax(control_outputs.logits, dim=1)[:, 1].detach().cpu().numpy()  # Probability of positive class

# Combine SST-2 and control predictions
all_preds = preds_sst2 + list(control_probs)

# Save predictions to BERT_FILE
output_df = pd.DataFrame({"pos": all_preds})
output_path = Config.BERT_FILE
output_df.to_csv(output_path, sep="\t", index=False, header=False)
print(f"Predictions saved to {output_path}")
