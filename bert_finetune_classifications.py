import torch
import pandas as pd
from datasets import Dataset
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from sklearn.preprocessing import LabelEncoder
import logging
import os

logging.basicConfig(level=logging.INFO)
os.environ["WANDB_DISABLED"] = "true"


class MultiClassSentimentModelV2:
    def __init__(self, model_name="microsoft/deberta-v3-small", csv_link=None):
        """Initialize model, tokenizer, and dataset."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load dataset and prepare labels
        self.df, self.label_map = self.__load_and_prepare_data(csv_link)
        self.num_labels = len(self.df["label"].unique())

        # Initialize model and configuration
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_labels
        ).to(self.device)
        self.dataset = self.__prepare_dataset(self.df)

    def __load_and_prepare_data(self, csv_link):
        """Load CSV and encode labels."""
        csv_link = (
            csv_link
            or "https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_multi_class_sentiment.csv"
        )
        logging.info(f"Loading dataset from {csv_link}")
        df = pd.read_csv(csv_link)

        # Encode labels to integers and store label mapping
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(
            df["label_name"]
        )  # Use 'label_name' for encoding
        label_map = dict(
            zip(range(len(label_encoder.classes_)), label_encoder.classes_)
        )  # Store label map
        return df, label_map  # Return both the dataframe and label map

    def __prepare_dataset(self, df):
        """Convert DataFrame to Hugging Face DatasetDict."""
        dataset = Dataset.from_pandas(df)
        dataset = dataset.train_test_split(test_size=0.2)
        return dataset.map(self.__tokenize, batched=True)

    def __tokenize(self, batch):
        """Tokenize text inputs."""
        return self.tokenizer(batch["text"], padding=True, truncation=True)

    def train_model(self, batch_size=16, epochs=2):
        """Train the model using Hugging Face Trainer."""
        training_args = TrainingArguments(
            output_dir="outputs",  # Where to save model checkpoints
            num_train_epochs=epochs,  # Number of epochs to train
            per_device_train_batch_size=batch_size,  # Batch size for training
            per_device_eval_batch_size=batch_size,  # Batch size for evaluation
            eval_strategy="epoch",  # Evaluate the model after each epoch
            save_strategy="epoch",  # Save the model if best performance
            learning_rate=5e-5,  # Learning rate
            weight_decay=0.01,  # Strength of weight decay
            logging_dir="./logs",  # Log every 10 steps
            load_best_model_at_end=True,  # Load best model at end of training
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self.__compute_metrics,
        )

        # Train the model
        trainer.train()

    @staticmethod
    def __compute_metrics(eval_pred):
        """Compute accuracy for evaluation."""
        from evaluate import load

        metric = load("accuracy")  # Define metric for evaluation
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return metric.compute(predictions=predictions, references=labels)

    def predict(self, text):
        """Make predictions using Hugging Face pipeline."""
        clf = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Get raw prediction result
        prediction = clf(text)[0]  # Get the first prediction (assuming single input)
        label_id = int(
            prediction["label"].split("_")[-1]
        )  # Extract numeric label from 'LABEL_0'

        # Map label ID to label_name
        label_name = self.label_map[label_id]
        prediction["label_name"] = label_name  # Add label_name to prediction
        return prediction

    def save_model(self, save_directory="save_model"):
        """Save the model to a specified directory."""
        from pathlib import Path

        SAVE_PATH = Path(save_directory)
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print(f"Model saved to directory: {save_directory}")

    def load_model(self, model_path="save_model"):
        """Load the model from a specified directory."""
        from pathlib import Path

        SAVE_PATH = Path(model_path)
        if not SAVE_PATH.exists():
            raise ValueError(f"The specified path does not exist: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


if __name__ == "__main__":
    model_path = "save_model"
    # Check if a saved model exists
    if os.path.exists(model_path):
        logging.info(f"Loading model from {model_path}")
        model = MultiClassSentimentModelV2()
        model.load_model(model_path)
    else:
        logging.info(f"No saved model found. Initializing and training a new model.")
        dataset, num_labels = load_and_prepare_data()
        model = MultiClassSentimentModelV2(num_labels=num_labels)
        model.initialize_model()
        model.train_model(dataset)

    app = gr.Interface(
        fn=model.predict,
        inputs=["text"],
        outputs=["text"],
        share=True,
    )
    app.launch()
