import argparse
from helpers import split_csv, combine_batches, preprocess_Da, preprocess_Do

def main(args):
    if args.action == 'split':
        split_csv(data_path="Experiment_data/Original_dataset" + args.data_name, input_file=args.input_file, data="Summary", label="Sentiment")
    elif args.action == 'combine':
        combine_batches(original_df_path="Experiment_data/Original_dataset" + args.data_name + "/SP-" + args.input_file, 
               translated_batch_path="Experiment_data/Original_dataset" + args.data_name + "/Translated_batch", 
               file_count=4, 
               output_path="Experiment_data/Original_dataset" + args.data_name + "/Translated-SP-" + args.input_file)
    elif args.action == 'preprocess_Da':
        preprocess_Da(data_path="Experiment_data/Original_dataset" + args.data_name, input_file="Translated-SP-" + args.input_file)
    elif args.action == 'preprocess_Do':
        preprocess_Do(data_path="Experiment_data/Original_dataset" + args.data_name, input_file=args.input_file, data=args.data, label=args.label)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Impacts")

    parser.add_argument('--action', type=str, default='split', choices=['split', 'combine', 'preprocess_Da', 'preprocess_Do'], help='Action to perform: train or inference or validation')
    
    # Split & combine
    parser.add_argument('--data_name', type=str, default='Flipkart', choices=['AIVIVN_2019', 'Flipkart', 'UIT_VSFC'], help='name of dataset')
    parser.add_argument('--input_file', type=str, default='Dataset-SA.csv', help='name of the file')
    parser.add_argument('--data', type=str, default='Summary')
    parser.add_argument('--label', type=str, default='Sentiment')

    # Preprocess
    args = parser.parse_args()
    main(args)