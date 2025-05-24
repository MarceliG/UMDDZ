from src import (
    analysis_dataset,
    evaluate_model,
    load_dataset_from_disk,
    load_skin_cancer_dataset,
    predict_random_sample,
    process_dataset,
    save_dataset_to_disk,
    train_model,
)
from utils import FilePath, logger, parse_args


def main() -> None:
    logger.info("Start application")
    args = parse_args()

    if args.download_dataset:
        logger.info("Downloading dataset...")
        dataset = load_skin_cancer_dataset()
        save_dataset_to_disk(dataset, FilePath.datasets_skin_cancer)

    dataset = load_dataset_from_disk(FilePath.datasets_skin_cancer)

    if args.analysis_dataset:
        logger.info("Analysis dataset...")
        analysis_dataset(dataset)

    if args.preprocessing:
        logger.info("Preprocessing dataset...")
        dataset = process_dataset(dataset)
        save_dataset_to_disk(dataset, FilePath.dataset_preprocessed_skin_cancer)

    if args.train:
        preprocessed_dataset = load_dataset_from_disk(FilePath.dataset_preprocessed_skin_cancer)
        logger.info("Training model...")
        train_model(preprocessed_dataset)

    if args.evaluate:
        logger.info("Making evaluation...")
        preprocessed_dataset = load_dataset_from_disk(FilePath.dataset_preprocessed_skin_cancer)
        evaluate_model(preprocessed_dataset)

    if args.predict:
        logger.info("Making predictions...")
        preprocessed_dataset = load_dataset_from_disk(FilePath.dataset_preprocessed_skin_cancer)
        predict_random_sample(preprocessed_dataset)

    logger.info("Finish application")


if __name__ == "__main__":
    main()
