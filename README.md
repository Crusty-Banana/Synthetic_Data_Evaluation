### Translate Flipkart data

1. split them into batches

```
python data_to_batches.py
```

2. send to chatGPT batches API to translate the reviews

3. download and combine batches from 


```
python main_data.py --action preprocess_Do --data_name AIVIVN_2019 --input_file train.csv --data comment --label label 
python main_data.py --action preprocess_Do --data_name AIVIVN_2019 --input_file test.csv --data comment --label label 
```