from transformers import (BertForSequenceClassification,
                          BertTokenizer)
import torch
from tqdm import tqdm  # Import tqdm

# Model Class
class CustomBERTModel:
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=2, device="cuda"):
        """Initialize the BERT model for classification.

        Args:
            model_name (str): Pre-trained BERT model name.
            num_labels (int): Number of output labels.
        """
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = device

    def train(self, train_dataloader, val_dataloader, epochs=3, learning_rate=5e-5):
        """Train the BERT model.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            # Wrap the train_dataloader with tqdm to show progress
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

            self.evaluate(val_dataloader, self.device)

    def evaluate(self, dataloader, device):
        """Evaluate the model on a validation dataset.

        Args:
            dataloader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to run the evaluation on.
        """
        self.model.eval()
        total, correct = 0, 0

        # Wrap the validation dataloader with tqdm to show progress
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

    def predict(self, text, max_length=128):
        """Perform inference on a single text.

        Args:
            dataloader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to run the evaluation on.
        """
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()

        return prediction
    
    def save_model(self, save_path):
        """Save the trained model to a file.

        Args:
            save_path (str): Path to save the model.
        """
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Load a model from a file.

        Args:
            load_path (str): Path to load the model from.
        """
        self.model = BertForSequenceClassification.from_pretrained(load_path)
        print(f"Model loaded from {load_path}")
