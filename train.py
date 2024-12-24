from models import CustomBERTModel
from datasets import CustomTextDataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import soft_preprocess_df

data_path = {
    "AIVIVN_2019": {
        "train": "data/AIVIVN_2019/train.csv",
        "test": "data/AIVIVN_2019/test.csv"
    },
    "Flipkart": {
        "train": "data/Flipkart/SP-Dataset.csv",
        "test": "data/Flipkart/SP-Dataset.csv"
    },
    "UIT-VSFC": {
        "train": "data/UIT-VSFC/train.csv",
        "test": "data/UIT-VSFC/test.csv"
    }
}

def train_model_with_dataset(model_name="bert-base-multilingual-cased", 
                             data_name="AIVIVN_2019", 
                             model_path="models/AIVIVN_2019_model"):
    """Train a model using a dataset.

    Args:
        model_name (string): Name of the model.
        data_name (string): Name of the dataset.
        model_path (string): Path to save the model.
    """
    train_csv = data_path[data_name]["train"]
    test_csv = data_path[data_name]["train"]

    train_data = soft_preprocess_df(pd.read_csv(train_csv), data="comment", label="label")
    test_data = soft_preprocess_df(pd.read_csv(test_csv), data="comment", label="label")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data['data'].tolist(),
        train_data['label'].tolist(),
        test_size=0.1,  # 10% validation split
        random_state=42
    )
    test_texts = test_data['data'].tolist()
    test_labels = test_data['label'].tolist()

    model = CustomBERTModel(model_name=model_name, num_labels=2)

    train_dataset = CustomTextDataset(texts=train_texts, labels=train_labels, tokenizer=model.tokenizer)
    val_dataset = CustomTextDataset(texts=val_texts, labels=val_labels, tokenizer=model.tokenizer)
    test_dataset = CustomTextDataset(texts=test_texts, labels=test_labels, tokenizer=model.tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.train(train_dataloader, val_dataloader, epochs=3, learning_rate=5e-5)

    new_text = "This is an amazing example."
    prediction = model.predict(new_text)
    print(f"Predicted label for '{new_text}': {prediction}")
    model.evaluate(test_dataloader, "cuda")

    model.save(model_path)