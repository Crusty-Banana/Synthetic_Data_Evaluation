import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
import json
import os

load_dotenv()

def soft_preprocess_df(df, data="Summary", label="Sentiment"):
    df = df[[data, label]].rename(columns={data: 'data', label: 'label'})

    label_encoder_sentiment = LabelEncoder()

    df['label'] = label_encoder_sentiment.fit_transform(df['label'])
    return df

def translation_request(idx, data):
    return {
                "custom_id": f"review-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": f"Translate review into Vietnamese, dont add meaning"},
                        {"role": "user", "content": f"{data}"}
                    ]
                }
            }

def split_into_batchs(df, output_path="data/Flipkart/To_be_translated_batch", batch_size=50000):
    file_count = 0
    line_count = 0
    
    os.makedirs(output_path, exist_ok=True) 
    batch_file = open(output_path + f"/batchinput_{file_count}.jsonl", "w")

    for idx, row in df.iterrows():
        json_line = translation_request(idx, row['data'])
        batch_file.write(json.dumps(json_line) + "\n")
        line_count += 1

        if line_count >= batch_size:
            batch_file.close()
            file_count += 1
            line_count = 0
            batch_file = open(output_path + f"/batchinput_{file_count}.jsonl", "w")

    batch_file.close()
    return file_count

def combine_batchs(original_df_path, translated_batch_path, file_count, output_path):
    translations = {}
    for i in range(file_count + 1):
        with open(translated_batch_path + f"/batch_{i}.jsonl", "r") as file:
            for line in file:
                response = json.loads(line)
                custom_id = response["custom_id"]
                translated_review = response["response"]["body"]["choices"][0]["message"]["content"]
                translations[custom_id] = translated_review

    df = pd.read_csv(original_df_path)
    df["Translated_Data"] = df.index.map(
        lambda idx: translations.get(f"review-{idx}", None)
    )

    df.to_csv(output_path, index=False)
    return df

def select_data_by_labels(df, labels=[0, 2]):
    df = df[df['labels'] in labels]
    df.loc[df['labels'] == 2, 'labels'] = 1
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}