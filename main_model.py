import argparse
from train import train_model_with_dataset
from models import CustomBERTModel
def main(args):
    if args.action == 'train':
        train_model_with_dataset(model_name=args.model_name, data_path=args.data_path, model_path=args.model_path, checkpoint_path=args.checkpoint_path)
        print("Training model {} on dataset {}".format(args.model_name, args.data_path))
    elif args.action == 'inference':
        model = CustomBERTModel()
        model.load_model(args.inference_model)

        prediction = model.predict(args.inference_text)

        print("Inference model {} on dataset {}:".format(args.model_name, args.data_path))
        print(f"Input: {args.inference_text}.\nOutput: {prediction}.")
    elif args.action == 'test':
        model = CustomBERTModel()
        model.load_model(args.inference_model)
        
        print("Validation model {} on dataset {}".format(args.model_name, args.data_path))
    
    elif args.action == 'data':
        pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Impacts")

    parser.add_argument('--action', type=str, default='train', choices=['train', 'inference', 'test', 'data'], help='Action to perform: train or inference or validation')
    
    # For Training
    parser.add_argument('--data_path', type=str, default='Experiment_data/Evaluation_dataset/AIVIVN.csv', help='Directory to load data')
    parser.add_argument('--model_name', type=str, default='mBERT', help='Name of the model')
    parser.add_argument('--model_path', type=str, default='', help='Directory to load trained model')
    parser.add_argument('--checkpoint_path', type=str, default='models/default_model', help='Directory to save trained model')

    # For Inference
    parser.add_argument('--inference_model', type=str, default="models/bert-base-multilingual-cased_AIVIVN_2019", help='Inference Text')
    parser.add_argument('--inference_text', type=str, default='sản phẩm bị lỗi', help='Inference Text')

    # For Testing
    parser.add_argument('--test_model', type=str, default="models/bert-base-multilingual-cased_AIVIVN_2019")
    parser.add_argument('--test_data_name', type=str, default='AIVIVN', choices=['AIVIVN', 'Flipkart', 'UIT-VSFC'])
    args = parser.parse_args()
    main(args)