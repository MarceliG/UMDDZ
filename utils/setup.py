import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset processing and model training")
    parser.add_argument("--download-dataset", "-d", action="store_true", help="Download dataset")
    parser.add_argument("--preprocessing", "-p", action="store_true", help="Preprocess dataset")
    parser.add_argument("--train", "-t", action="store_true", help="Train model")
    parser.add_argument("--predict", "-pred", action="store_true", help="Make predictions")
    return parser.parse_args()
