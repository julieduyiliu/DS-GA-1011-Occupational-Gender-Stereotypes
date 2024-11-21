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

model = RobertaForSequenceClassification.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# testing on new sentences
sentences = utils.get_sentences()
if isinstance(sentences, np.ndarray):
    sentences = sentences.tolist()

assert all(isinstance(sentence, str) for sentence in sentences), "All elements must be strings!"
tokenized_sentences = tokenizer(sentences, padding="max_length", truncation=True, max_length=Config.MAX_SEQUENCE_LENGTH, return_tensors="pt")

with torch.no_grad():
    outputs = model(
        input_ids=tokenized_sentences["input_ids"],
        attention_mask=tokenized_sentences["attention_mask"]
    )
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)
female_probs = probabilities[:, 1]  
male_probs = probabilities[:, 0]  

female_probs = female_probs.cpu().numpy()
male_probs = male_probs.cpu().numpy()

n = len(male_probs)
preds = np.concatenate([male_probs, female_probs])
# preds = torch.argmax(probabilities, dim=-1)
(t, prob, diff) = utils.ttest(preds)
print(f'Gender Differences: {diff}')

# bias scores

mean_male_prob = np.mean(male_probs)
mean_female_prob = np.mean(female_probs)

bias_score = abs(mean_male_prob - mean_female_prob)
print(f"Bias Score: {bias_score}")
