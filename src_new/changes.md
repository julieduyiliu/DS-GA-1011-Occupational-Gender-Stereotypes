Here's a concise summary of the changes I made to the original source code:
1. Added BERT Fine-tuning: introduced fine-tuning for a pretrained BERT model (BERT-Base, uncased) on the SST-2 dataset, including data loading, tokenization, and setting up a Trainer with parameters like batch size, learning rate, epochs, and logging options. Predictions were saved as positive class probabilities compatible with analysis.py.
2. Model Format Recommendation: Updated the model-saving format to .keras over .h5 in response to TensorFlow/Keras recommendations for better compatibility and future-proofing.
3. Data Type Update: Replaced np.float with float to align with recent NumPy standards, avoiding deprecation warnings and ensuring compatibility with newer libraries.
4. Changes made in analysis.py
  - Modified analysis.py to read positive class probabilities from the BERT_FILE for direct use in statistical testing and gender-based mean comparisons.
  - Added checks to prevent NaN outputs by ensuring statistical tests are performed on valid arrays and handling cases where data might be missing or incomplete.
  - Adjusted variable names and read functions to match the new BERT file structure, ensuring consistent access to required data.
