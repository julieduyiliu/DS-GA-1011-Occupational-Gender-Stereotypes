#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# bert_finetune.py

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
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

# Initialize tokenizer and BERT model with hidden states enabled
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, num_labels=2)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)

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

# Dataset class for PyTorch
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

# Define the adversarial discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize discriminator
discriminator = Discriminator(input_size=768)  # Hidden size of BERT's CLS token
discriminator.to(device)

# Optimizers and loss functions
optimizer = optim.AdamW(list(model.parameters()) + list(discriminator.parameters()), lr=2e-5)
classification_loss_fn = nn.CrossEntropyLoss()
adversarial_loss_fn = nn.BCELoss()

# Training function
def train_model(train_dataset, dev_dataset, epochs=3):
    model.train()
    discriminator.train()

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=8)

    for epoch in range(epochs):
        total_class_loss, total_adv_loss = 0, 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Extract input tensors
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass through BERT
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1][:, 0, :]  # CLS token embeddings

            # Compute classification loss
            class_loss = classification_loss_fn(logits, labels)

            # Forward pass through discriminator for adversarial loss
            adv_logits = discriminator(hidden_states)
            adv_labels = torch.full_like(adv_logits, 0.5)  # Neutral target for debiasing
            adv_loss = adversarial_loss_fn(adv_logits.squeeze(), adv_labels.squeeze())

            # Combine losses
            loss = class_loss + adv_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_class_loss += class_loss.item()
            total_adv_loss += adv_loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Classification Loss: {total_class_loss:.4f} | Adversarial Loss: {total_adv_loss:.4f}")

    # Evaluate model
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")

# Train the model
train_model(train_dataset, dev_dataset, epochs=3)

# Save the model and discriminator
torch.save(model.state_dict(), Config.BERT_MODEL_FILE)
torch.save(discriminator.state_dict(), Config.DISCRIMINATOR_FILE)
print(f"Model and discriminator saved to {Config.BERT_MODEL_FILE} and {Config.DISCRIMINATOR_FILE}")

# Generate predictions and save as BERT_FILE
model.eval()
with torch.no_grad():
    dev_loader = DataLoader(dev_dataset, batch_size=8)
    dev_logits = []
    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        dev_logits.extend(outputs.logits.cpu().numpy())

dev_probs = torch.softmax(torch.tensor(dev_logits), dim=1)[:, 1].numpy()

# Add control predictions
control_sentences = ["This man is a person.", "This woman is a person."] * 20
control_encodings = tokenizer(control_sentences, truncation=True, padding=True, max_length=Config.MAX_SEQUENCE_LENGTH, return_tensors="pt").to(device)

with torch.no_grad():
    control_outputs = model(**control_encodings)
control_probs = torch.softmax(control_outputs.logits, dim=1)[:, 1].cpu().numpy()

# Save predictions
all_preds = list(dev_probs) + list(control_probs)
output_df = pd.DataFrame({"pos": all_preds})
output_path = Config.BERT_FILE
output_df.to_csv(output_path, sep="\t", index=False, header=False)
print(f"Predictions saved to {output_path}")