import pandas as pd

df = pd.read_csv('Experiment_data/Augmented_dataset/Flipkart.csv')

non_str_data = df[~df['data'].apply(lambda x: isinstance(x, str))]
if non_str_data.empty:
    print("All data in 'data' column is of type str")
else:
    print("The following rows have non-string data in 'data' column:")
    print(non_str_data)