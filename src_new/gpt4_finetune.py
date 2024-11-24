from config import Config
SEED = Config.SEED

import numpy as np
np.random.seed(SEED)
import pandas as pd
import logging
import json
import time
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tiktoken
import utils


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load data
train = pd.read_csv(Config.TSV_TRAIN, sep="\t", header=None, names=["idx", "class", "dummy", "text"])
dev = pd.read_csv(Config.TSV_DEV, sep="\t", header=None, names=["idx", "class", "dummy", "text"])
X_train, X_dev, y_train, y_dev = train["text"], dev["text"], train["class"], dev["class"]


class GPTFineTuner:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 4096
        self.model_name = "gpt-4o-mini-2024-07-18"
        
    def prepare_data(self, X, y):
       """Convert training data to OpenAI format"""
       system_prompt = """You are a sentiment analysis model.
                        Task: Classify text sentiment as negative (0) or positive (1).
                        Rules:
                        1. Provide only binary output: 0 or 1
                        2. Consider:
                            - Language tone
                            - Emotional content
                            - Context clues
                        3. Ignore neutral statements
                        4. No explanation needed
                        5. Focus on overall sentiment"""
       formatted_data = []
       for text, label in zip(X, y):
           if text.strip():  # Check if the text is not empty
               messages = [
                   {"role": "system", "content": system_prompt},
                   {"role": "user", "content": text},
                   {"role": "assistant", "content": str(label)}
               ]
               formatted_data.append({"messages": messages})
           else:
               logger.warning("Empty text entry found and skipped.")  # Log warning for empty text
           
       return formatted_data

    def validate_data(self, data):
       """Validate token counts"""
       for item in data:
           total_tokens = 0
           for message in item['messages']:
               tokens = self.tokenizer.encode(message['content'])
               total_tokens += len(tokens)
               
           if total_tokens > self.max_tokens:
               logger.warning(f"Sample exceeds token limit: {total_tokens} tokens")
               return False
       return True
       

    def save_training_file(self, data, filename):
        """Save as JSONL"""
        with open(filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Training data saved to {filename}")


    def create_fine_tuning_job(self, training_file):
       """Create and monitor fine-tuning"""
       try:
           file = self.client.files.create(
               file=open(training_file, 'rb'),
               purpose='fine-tune'
           )
           
           job = self.client.fine_tuning.jobs.create(
               training_file=file.id,
               model=self.model_name,
               hyperparameters={"n_epochs": 3}
           )
           
           while True:
               status = self.client.fine_tuning.jobs.retrieve(job.id)
               logger.info(f"Status: {status.status}")
               if status.status in ['succeeded', 'failed']:
                   break
               time.sleep(60)

           return status

       except Exception as e:
           logger.error(f"Fine-tuning error: {str(e)}")
           raise

    def get_predictions(self, model_id, data):
        """Get predictions from model"""
        predictions = []
        for item in data:
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=item['messages'][:-1] if len(item['messages']) > 2 else item['messages'],
                    temperature=0
                )
                pred = float(response.choices[0].message.content)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                continue
        return predictions

def main():
    # Load data
    train = pd.read_csv(Config.TSV_TRAIN, sep="\t", header=None, 
                       names=["idx", "class", "dummy", "text"])
    dev = pd.read_csv(Config.TSV_DEV, sep="\t", header=None, 
                     names=["idx", "class", "dummy", "text"])
    
    X_train, X_dev = train["text"], dev["text"]
    y_train, y_dev = train["class"], dev["class"]
    
    # Initialize fine-tuner
    fine_tuner = GPTFineTuner(api_key=Config.OPENAI_API_KEY)
    logger.info("Starting fine-tuning process")

    # Prepare training data
    train_data = fine_tuner.prepare_data(X_train, y_train)
    dev_data = fine_tuner.prepare_data(X_dev, y_dev)

    # Validate data
    if not fine_tuner.validate_data(train_data):
        logger.error("Data validation failed")
        return

    # Save training file
    training_file = Config.JSONL_TRAIN
    fine_tuner.save_training_file(train_data, training_file)

    # Start fine-tuning
    job_status = fine_tuner.create_fine_tuning_job(training_file)
    
    if job_status.status == 'succeeded':
        # Save model ID
        with open(Config.GPT_MODEL_FILE, 'w') as f:
            json.dump({'model_id': job_status.fine_tuned_model}, f)
        
        # Get dev set predictions
        dev_predictions = fine_tuner.get_predictions(
            job_status.fine_tuned_model, 
            dev_data
        )
        dev_accuracy = accuracy_score(y_dev, [int(p >= 0.5) for p in dev_predictions])
        print(f"Dev accuracy: {dev_accuracy}")

        # Prepare test sentences
        sentences = utils.get_sentences()
        test_data = fine_tuner.prepare_data(
            sentences, 
            ['0'] * len(sentences)  # Dummy labels
        )
        
        # Get test predictions
        test_predictions = fine_tuner.get_predictions(
            job_status.fine_tuned_model,
            test_data
        )
        
        # Run statistical tests
        (t, prob, diff) = utils.ttest(test_predictions)
        
        # Save predictions with dev accuracy in header
        file = Config.GPT_FILE
        np.savetxt(
            file, 
            test_predictions,
            header=f"dev accuracy: {dev_accuracy}"
        )
        
        logger.info(f"Results saved to {Config.GPT_FILE}")
        logger.info(f"Final dev accuracy: {dev_accuracy}")
        logger.info(f"T-test results - t: {t}, p: {prob}, diff: {diff}")

if __name__ == "__main__":
    main()