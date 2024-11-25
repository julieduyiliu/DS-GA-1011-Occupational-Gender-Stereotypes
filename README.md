# Why albert-base-v2?

albert-base-v2 has approximately 12M parameters, making it closer in size to BERT-base (110M parameters) than smaller ALBERT variants like albert-small or albert-tiny. The v2 versions of ALBERT (e.g., albert-base-v2) fix known issues in v1, such as sentence order prediction inefficiencies, making it better suited for comparison.
---
1. **Dataset Prep**  
   - `generate_corpus.py` generates a `gender.tsv` dataset for gender bias testing by combining male and female nouns with professions (e.g., "This man is a doctor.").  
   - The SST-2 dataset (`train.tsv` and `dev.tsv`) is used for sentiment analysis tasks, containing labeled sentences for training and validation.  
   - Paths to these datasets are managed in `config.py`.

2. **Model Baselines**  
   - **Logistic Regression (`baseline.py`)**:
     - Loads `train.tsv` and `dev.tsv` for training and validation.
     - Prepares Bag-of-Words and TF-IDF features from text data.
     - Trains a logistic regression model, achieving baseline performance (e.g., 85% accuracy).
     - Outputs predictions for `gender.tsv` to `Config.LOGREG_FILE`.

   - **BiLSTM Model (`bilstm.py`)**:
     - Uses GloVe embeddings for initializing a bidirectional LSTM.
     - Trains and evaluates the model on SST-2 data, achieving improved performance (e.g., 88% accuracy).
     - Outputs predictions for `gender.tsv` to `Config.LSTM_FILE`.

3. **Fine-Tuning ALBERT (`albert_finetune.py`)**  
   - Tokenizes sentences from `train.tsv`, `dev.tsv`, and `gender.tsv` using the ALBERT tokenizer.
   - Fine-tunes ALBERT for binary classification tasks, achieving high accuracy (e.g., 90%).
   - Generates predictions for `gender.tsv` (e.g., "This man is a doctor." -> 0.92, "This woman is a doctor." -> 0.88) and saves them to `Config.ALBERT_FILE`.

4. **Analysis of Gender Bias (`analysis.py`)**  
   - Compares male vs. female predictions for professions in `gender.tsv` using t-tests (e.g., females receive slightly lower positive sentiment scores).  
   - Evaluates gender disparities in SST-2, analyzing male/female-associated sentences for sentiment differences.  
   - Plots the relationship between predicted sentiment probabilities and real-world metrics (e.g., median earnings for professions).  

5. **Configuration and Utility Management**  
   - `config.py` serves as the backbone, centralizing file paths, constants, and experiment-specific values.  
   - `utils.py` provides shared functionality like GloVe embedding loading, sentence extraction, and statistical testing.  


