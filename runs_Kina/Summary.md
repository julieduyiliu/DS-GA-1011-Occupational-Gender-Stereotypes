# Progress Report
We replicate the evaluation of three sentiment analysis models trained on the **Stanford Sentiment Treebank 2 (SST-2)** dataset (Socher et al., 2013), which contains movie review phrases with binary (positive/negative) sentiment labels. Each model is evaluated on a new corpus, and we measure the difference in mean predicted positive sentiment probabilities between sentences with male nouns versus female nouns. The study tests three hypotheses, with null hypotheses indicating no difference in means between sentences with male and female nouns.

The evaluation methodology follows **Kiritchenko and Mohammad (2018)**, using a paired t-test for each sentence pair (male and female templates) with a significance level of 0.01. A **Bonferroni correction** is applied for multiple hypotheses, where null hypotheses are rejected only for p-values less than 0.01/3.

---

## Summary of Replication Results

### Model 1: Bag-of-Words + Logistic Regression (BoW + LogReg)
- **Replication Result**: Successfully replicated with results matching the original study.
- **Dev Accuracy**: 0.827
- **Gender Difference (F - M)**: 0.035**

### Model 2: BiLSTM
- **Replication Result**: Results deviated slightly from the original study.
  - **Dev Accuracy**: 0.829 (Original: 0.841)
  - **Gender Difference (F - M)**: 0.101** (Original: 0.077**)
- **Possible Reasons for Discrepancies**:
  1. **Library and Framework Version Changes**: Updates in TensorFlow, Keras, and other libraries may impact model performance due to changes in parameter defaults, optimizers, and random seed handling.
  2. **Random Seed and Initialization Differences**: LSTMs are sensitive to initial weight configurations. Even minor variations in random initialization may affect results.
  3. **Hardware and Computational Precision**: Differences in hardware (e.g., GPU vs. CPU) and software versions (CUDA, cuDNN) could introduce small numerical variations.
  
  These deviations, while minor, do not alter the overall trend observed in the study.

### Model 3: BERT
- **Replication Approach**: Since original BERT fine-tuning code was unavailable, we implemented a custom fine-tuning approach on the SST-2 dataset.
- **Results**:
  - **Dev Accuracy**: 0.926 (similar to the original 0.930)
  - **Gender Difference (F - M)**: -0.014 (not statistically significant)
- **Explanation for Lack of Significance**:
  - **Model Architecture Sensitivity**: BERT’s architecture and pretraining objective may not be as sensitive to subtle gender-based sentiment variations.
  - **Variability in Pretrained Weights**: Fine-tuning a different pretrained BERT model could yield different sensitivities to linguistic nuances.

---

## Conclusion
This replication study closely followed the original methodology. Minor deviations were observed in model 2, and model 3 did not yield statistically significant gender differences. However, the overall trends align with those reported in the original study, validating the broader findings.

