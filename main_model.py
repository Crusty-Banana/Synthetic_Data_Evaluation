import argparse
from train import train_model_with_dataset
from models import CustomBERTModel
def main(args):
    if args.action == 'train-new':
        train_model_with_dataset(model_name=args.model_name, data_name=args.data_name, model_path=args.model_dir)
        print("Training model {} on dataset {}".format(args.model_name, args.data_name))
    elif args.action == 'train-old':
        pass
    elif args.action == 'inference':
        model = CustomBERTModel()
        model.load_model(args.inference_model)

        prediction = model.predict(args.inference_text)

        print("Inference model {} on dataset {}:".format(args.model_name, args.data_name))
        print(f"Input: {args.inference_text}.\nOutput: {prediction}.")
    elif args.action == 'test':
        model = CustomBERTModel()
        model.load_model(args.inference_model)
        
        print("Validation model {} on dataset {}".format(args.model_name, args.data_name))
    
    elif args.action == 'data':
        pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Impacts")

    parser.add_argument('--action', type=str, default='train', choices=['train-new', 'train-old', 'inference', 'test', 'data'], help='Action to perform: train or inference or validation')
    
    # For Training
    parser.add_argument('--data_name', type=str, default='AIVIVN', choices=['AIVIVN', 'Flipkart', 'UIT-VSFC'], help='Local directory for dataset')
    parser.add_argument('--model_name', type=str, default='mBERT')
    parser.add_argument('--model_path', type=str, default='models/bert-base-multilingual-cased_AIVIVN_2019', help='Directory to load trained model')
    parser.add_argument('--checkpoint_path', type=str, default='models', help='Directory to save trained model', help='Directory to save trained model')

    # For Inference
    parser.add_argument('--inference_model', type=str, default="models/bert-base-multilingual-cased_AIVIVN_2019", help='Inference Text')
    parser.add_argument('--inference_text', type=str, default='sản phẩm bị lỗi', help='Inference Text')

    # For Testing
    parser.add_argument('--test_model', type=str, default="models/bert-base-multilingual-cased_AIVIVN_2019")
    parser.add_argument('--test_data_name', type=str, default='AIVIVN', choices=['AIVIVN', 'Flipkart', 'UIT-VSFC'])
    args = parser.parse_args()
    main(args)