import os

import torch
from datasets import DatasetDict, load_dataset
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import ViTImageProcessor
from utils import FilePath


def process_dataset(dataset: DatasetDict) -> DatasetDict:
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    categorical_columns = ["dx", "dx_type", "sex", "localization"]
    label_encoders = {col: LabelEncoder().fit(dataset["train"][col]) for col in categorical_columns}

    age_mean = sum(filter(None, dataset["train"]["age"])) / len(dataset["train"]["age"])
    age_std = (
        sum((x - age_mean) ** 2 for x in filter(None, dataset["train"]["age"])) / len(dataset["train"]["age"])
    ) ** 0.5

    def transform(example) -> DatasetDict:
        image = example["image"].convert("RGB")
        transforms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ])
        example["pixel_values"] = transforms(image)

        example["metadata"] = torch.tensor(
            [
                *[label_encoders[col].transform([example[col]])[0] for col in categorical_columns],
                (example["age"] - age_mean) / age_std if example["age"] is not None else 0.0,
            ],
            dtype=torch.float32,
        )
        example["label"] = label_encoders["dx"].transform([example["dx"]])[0]
        return example

    dataset = dataset.map(
        transform, remove_columns=["image", "image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization"]
    )
    dataset.save_to_disk(FilePath.dataset_preprocessed_skin_cancer)
    return dataset


def load_skin_cancer_dataset() -> DatasetDict:
    return load_dataset("marmal88/skin_cancer")


def save_dataset_to_disk(dataset: DatasetDict, path: str) -> None:
    dataset.save_to_disk(path)


def load_dataset_from_disk(path: str) -> DatasetDict:
    if os.path.exists(path):
        return DatasetDict.load_from_disk(path)
    error_message = f"The directory {path} does not exist."
    raise FileNotFoundError(error_message)
