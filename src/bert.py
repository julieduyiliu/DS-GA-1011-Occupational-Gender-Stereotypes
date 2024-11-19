## fine-tune a pretrained (uncased) BERT-Base model on the SST-2 dataset
from config import Config
SEED = Config.SEED

import numpy as np
np.random.seed(SEED)
from tensorflow.random import set_seed
set_seed(SEED)
import pandas as pd
import utils

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from datasets import Dataset

##########################################

# Load the SST-2 dataset
train = pd.read_csv(Config.TSV_TRAIN, sep="\t", header=None, names=["idx", "class", "dummy", "text"])
dev = pd.read_csv(Config.TSV_DEV, sep="\t", header=None, names=["idx", "class", "dummy", "text"])
X_train, X_dev, y_train, y_dev = train["text"], dev["text"], train["class"], dev["class"]



# Load the pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the dataset
def tokenize(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=Config.MAX_SEQUENCE_LENGTH)

# Create Dataset objects
train_dataset = Dataset.from_dict({
    'text': X_train.tolist(),
    'label': y_train.tolist()
})

dev_dataset = Dataset.from_dict({
    'text': X_dev.tolist(),
    'label': y_dev.tolist()
})

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)

# Prepare the datasets for Trainer
train_dataset = train_dataset.rename_column("label", "labels")
dev_dataset = dev_dataset.rename_column("label", "labels")

# Set up the training arguments
training_args = TrainingArguments(
    output_dir=str(Config.BERT_FILE),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=str(Config.BERT_FILE),
    logging_steps=10,
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained(str(Config.BERT_FILE.parent))
tokenizer.save_pretrained(str(Config.BERT_FILE.parent))

# Evaluate the model
dev_preds = trainer.predict(dev_dataset)
pred_labels = np.argmax(dev_preds.predictions, axis=1)
print("Validation Accuracy:", accuracy_score(y_dev, pred_labels))

# Example for getting predictions on new sentences
sentences = utils.get_sentences()
tokenized_sentences = tokenizer(sentences, padding="max_length", truncation=True, max_length=Config.MAX_SEQUENCE_LENGTH, return_tensors="pt")
preds = model.predict(tokenized_sentences['input_ids'], attention_mask=tokenized_sentences['attention_mask'])
pred_labels = np.argmax(preds.logits, axis=1)

# Save predictions to a file
file = Config.BERT_FILE
np.savetxt(file, pred_labels, header="dev accuracy: " + str(accuracy_score(y_dev, pred_labels)))
