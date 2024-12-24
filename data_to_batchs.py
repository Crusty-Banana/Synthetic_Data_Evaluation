from helpers import soft_preprocess_csv, split_into_batchs
import pandas as pd

df = pd.read_csv("data/Flipkart/Dataset-SA.csv")
df = soft_preprocess_csv(df, 
                         data="Summary", 
                         label="Sentiment")
df.to_csv("data/Flipkart/SP-Dataset.csv")

split_into_batchs(df=df, 
                  translated_batch_path="data/Flipkart/Translated_batch",
                  output_path="data/Flipkart/To_be_translated_batch", 
                  batch_size=50000)