from config import Config
SEED = Config.SEED

import numpy as np
import torch
np.random.seed(SEED)
from tensorflow.random import set_seed
set_seed(SEED)
import pandas as pd
import utils 

from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from datasets import Dataset

# load SST-2 dataset
train = pd.read_csv(Config.TSV_TRAIN, sep='\t', header = None, names = ['idx', 'class', 'dummy', 'text'])
dev = pd.read_csv(Config.TSV_DEV, sep='\t', header = None, names = ['idx', 'class', 'dummu', 'text'])
X_train, X_dev, y_train, y_dev = train['text'], dev['text'], train['class'], dev['class']

# load the pretrained RoBERTa and tokenizer
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(examples): 
    return tokenizer(examples['text'], padding = 'max_length', truncation = True, max_length = Config.MAX_SEQUENCE_LENGTH)

train_dataset = Dataset.from_dict({
    'text': X_train.tolist(), 
    'label': y_train.tolist() 
})

dev_dataset = Dataset.from_dict({
    'text': X_dev.tolist(), 
    'label': y_dev.tolist()
})

train_dataset_tok = train_dataset.map(tokenize, batched = True)
dev_dataset_tok = dev_dataset.map(tokenize, batched = True)

train_dataset_tok = train_dataset_tok.rename_column('label', 'labels')
dev_dataset_tok = dev_dataset_tok.rename_column('label', 'labels')

# set up training
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=3, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    warmup_steps=500, 
    warmup_ratio=0.01, 
    logging_dir='./results',
    logging_steps = 10
)

trainer = Trainer(
    model = model, 
    args = training_args, 
    train_dataset=train_dataset_tok, 
    eval_dataset=  dev_dataset_tok
)

trainer.train()

model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

dev_preds = trainer.predict(dev_dataset_tok)
pred_labels = np.argmax(dev_preds.predictions, axis = 1)
print('Validation Accuracy:', accuracy_score(y_dev, pred_labels))

