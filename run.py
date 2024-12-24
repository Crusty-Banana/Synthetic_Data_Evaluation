from models import CustomBERTModel
from datasets import CustomTextDataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import soft_preprocess_df

train_csv = "data/AIVIVN_2019/train.csv"  # Replace with your train dataset path
test_csv = "data/AIVIVN_2019/test.csv"    # Replace with your test dataset path

train_data = soft_preprocess_df(pd.read_csv(train_csv), data="comment", label="label")
test_data = soft_preprocess_df(pd.read_csv(test_csv), data="comment", label="label")

# Split train_data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_data['data'].tolist(),
    train_data['label'].tolist(),
    test_size=0.1,  # 10% validation split
    random_state=42
)

# Prepare the test set
test_texts = test_data['data'].tolist()
test_labels = test_data['label'].tolist()

# Initialize the Model
model = CustomBERTModel(model_name='bert-base-multilingual-cased', num_labels=2)

# Create Datasets
train_dataset = CustomTextDataset(texts=train_texts, labels=train_labels, tokenizer=model.tokenizer)
val_dataset = CustomTextDataset(texts=val_texts, labels=val_labels, tokenizer=model.tokenizer)
test_dataset = CustomTextDataset(texts=test_texts, labels=test_labels, tokenizer=model.tokenizer)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# For simplicity, use the same data for validation in this example
val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# Train the Model
model.train(train_dataloader, val_dataloader, epochs=3, learning_rate=5e-5)

# Example Inference
new_text = "This is an amazing example."
prediction = model.predict(new_text)
print(f"Predicted label for '{new_text}': {prediction}")
