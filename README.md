## Prepare Data

### Prepare $D_a$ (Augmented Data)

1. Split them into batches

```
python main_data.py --action split --data_name Flipkart --input_file Dataset-SA.csv --data Summary --label Sentiment
```

2. Send to chatGPT batches API to translate the reviews

Batches are in "Experiment_data/Original_dataset/Flipkart/To_be_translated_batch"

3. Download and combine batches from chatGPT.

Put batches translated from chatGPT with name batch_{id}.jsonl in "Experiment_data/Original_dataset/Flipkart/Translated_batch"

```
python main_data.py --action combine --data_name Flipkart --input_file Dataset-SA.csv
```

4. Preprocess translated Data.

```
python main_data.py --action preprocess_Da --data_name Flipkart --input_file Dataset-SA.csv
```

Result is "Experiment_data/Original_dataset/Flipkart/Translated-SP-Dataset-SA.csv"

Change name to Flipkart.csv and move to "Experiment_data/Augmented_dataset"

5. Sample translated Data.

```
python main_data.py --action sample --data_path Experiment_data/Augmented_dataset/Flipkart.csv --output_path Experiment_data/Augmented_dataset/Flipkart10.csv --percent_sample_size 10


python main_data.py --action sample --data_path Experiment_data/Augmented_dataset/Flipkart.csv --output_path Experiment_data/Augmented_dataset/Flipkart30.csv --percent_sample_size 30


python main_data.py --action sample --data_path Experiment_data/Augmented_dataset/Flipkart.csv --output_path Experiment_data/Augmented_dataset/Flipkart60.csv --percent_sample_size 60


python main_data.py --action sample --data_path Experiment_data/Augmented_dataset/Flipkart.csv --output_path Experiment_data/Augmented_dataset/Flipkart100.csv --percent_sample_size 100
```

### Prepare $D_c$ (Control Data)

```
python main_data.py --action preprocess_Do --data_name AIVIVN_2019 --input_file train.csv --data comment --label label 
```

Result is "Experiment_data/Original_dataset/AIVIVN_2019/PP-train.csv"

Change name to AIVIVN.csv and move to "Experiment_data/Control_dataset"


1. Concate $D_c$ + $D_a$

```
python main_data.py --action concate --data_path1 Experiment_data/Control_dataset/AIVIVN.csv --data_path2 Experiment_data/Augmented_dataset/Flipkart10.csv --output_path Experiment_data/Augmented_dataset/AIVIVN+Flipkart10.csv

python main_data.py --action concate --data_path1 Experiment_data/Control_dataset/AIVIVN.csv --data_path2 Experiment_data/Augmented_dataset/Flipkart30.csv --output_path Experiment_data/Augmented_dataset/AIVIVN+Flipkart30.csv

python main_data.py --action concate --data_path1 Experiment_data/Control_dataset/AIVIVN.csv --data_path2 Experiment_data/Augmented_dataset/Flipkart60.csv --output_path Experiment_data/Augmented_dataset/AIVIVN+Flipkart60.csv

python main_data.py --action concate --data_path1 Experiment_data/Control_dataset/AIVIVN.csv --data_path2 Experiment_data/Augmented_dataset/Flipkart100.csv --output_path Experiment_data/Augmented_dataset/AIVIVN+Flipkart100.csv
```

### Prepare Evaluation Data

```
python main_data.py --action preprocess_Do --data_name AIVIVN_2019 --input_file test.csv --data comment --label label 
```

Result is "Experiment_data/Original_dataset/AIVIVN_2019/PP-test.csv"

Change name to AIVIVN.csv and move to "Experiment_data/Evaluation_dataset"

## Experiment

### mBERT on $D_c$
```
python main_model.py --action train --data_path Experiment_data/Control_dataset/AIVIVN.csv --model_name mBERT --checkpoint_path models/mBert_Control
```

### mBERT on $D_c$ then $D_a$

```
python main_model.py --action train --data_path Experiment_data/Augmented_dataset/Flipkart10.csv --model_name mBERT --checkpoint_path models/mBert_ControlThenAugment10 --model_path models/mBert_Control
```

### mBERT on $D_c$ + $D_a$

```
python main_model.py --action train --data_path Experiment_data/Augmented_dataset/AIVIVN+Flipkart10.csv --model_name mBERT --checkpoint_path models/mBert_Control+Augment10

python main_model.py --action train --data_path Experiment_data/Augmented_dataset/AIVIVN+Flipkart30.csv --model_name mBERT --checkpoint_path models/mBert_Control+Augment30

python main_model.py --action train --data_path Experiment_data/Augmented_dataset/AIVIVN+Flipkart60.csv --model_name mBERT --checkpoint_path models/mBert_Control+Augment60

python main_model.py --action train --data_path Experiment_data/Augmented_dataset/AIVIVN+Flipkart100.csv --model_name mBERT --checkpoint_path models/mBert_Control+Augment100
```

## Evaluate

```
python main_model.py --action test --model_name mBert_Control --test_model models/mBert_Control --test_data_path Experiment_data/Evaluation_dataset/AIVIVN.csv

python main_model.py --action test --model_name mBert_ControlThenAugment10 --test_model models/mBert_ControlThenAugment10 --test_data_path Experiment_data/Evaluation_dataset/AIVIVN.csv

python main_model.py --action test --model_name mBert_Control+Augment10 --test_model models/mBert_Control+Augment10 --test_data_path Experiment_data/Evaluation_dataset/AIVIVN.csv

python main_model.py --action test --model_name mBert_Control+Augment30 --test_model models/mBert_Control+Augment30 --test_data_path Experiment_data/Evaluation_dataset/AIVIVN.csv

python main_model.py --action test --model_name mBert_Control+Augment60 --test_model models/mBert_Control+Augment60 --test_data_path Experiment_data/Evaluation_dataset/AIVIVN.csv

python main_model.py --action test --model_name mBert_Control+Augment100 --test_model models/mBert_Control+Augment100 --test_data_path Experiment_data/Evaluation_dataset/AIVIVN.csv
```
### TODO:

Better evaluation
Hyper param tune model
add MLP layer at the end of the model
