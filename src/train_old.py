import os

import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import Trainer, TrainingArguments, ViTImageProcessor, ViTModel

DATASET_PATH = "skin_cancer"
PROCESSED_DATASET_PATH = "processed_skin_cancer"
MODEL_SAVE_PATH = "vit_skin_cancer_model"

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)


def load_or_process_dataset():
    if os.path.exists(PROCESSED_DATASET_PATH):
        return load_from_disk(PROCESSED_DATASET_PATH)

    dataset = load_from_disk(DATASET_PATH)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    categorical_columns = ["dx", "dx_type", "sex", "localization"]
    numerical_columns = ["age"]
    label_encoders = {col: LabelEncoder().fit(dataset["train"][col]) for col in categorical_columns}

    age_mean = sum(filter(None, dataset["train"]["age"])) / len(dataset["train"]["age"])
    age_std = (
        sum((x - age_mean) ** 2 for x in filter(None, dataset["train"]["age"])) / len(dataset["train"]["age"])
    ) ** 0.5

    def transform(example):
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
    dataset.save_to_disk(PROCESSED_DATASET_PATH)
    return dataset


def collate_fn(batch):
    pixel_values = torch.stack([torch.tensor(item["pixel_values"]) for item in batch])
    metadata = torch.stack([torch.tensor(item["metadata"]) for item in batch])  # Wymuszenie konwersji
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "metadata": metadata, "labels": labels}


class MultiModalClassifier(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.metadata_fc = nn.Linear(num_metadata_features, 128)
        self.classifier = nn.Linear(self.vit.config.hidden_size + 128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, pixel_values, metadata, labels=None):
        vit_output = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        metadata_output = self.relu(self.metadata_fc(metadata))
        combined = torch.cat((vit_output, metadata_output), dim=1)
        logits = self.classifier(combined)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return logits


def main():
    dataset = load_or_process_dataset()
    num_metadata_features = len(["dx", "dx_type", "sex", "localization"]) + 1
    num_labels = len(set(dataset["train"]["label"]))

    model = MultiModalClassifier(num_metadata_features, num_labels)

    training_args = TrainingArguments(
        output_dir="./vit_skin_cancer",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        dataloader_num_workers=8,
        gradient_accumulation_steps=2,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
    )

    trainer.train()
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "model.pth"))


def eval_model():
    dataset = load_or_process_dataset()
    num_metadata_features = len(["dx", "dx_type", "sex", "localization"]) + 1
    num_labels = len(set(dataset["train"]["label"]))

    model = MultiModalClassifier(num_metadata_features, num_labels)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "model.pth")))
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for sample in dataset["test"]:
            pixel_values = torch.tensor(sample["pixel_values"], dtype=torch.float32).unsqueeze(0)
            metadata = torch.tensor(sample["metadata"], dtype=torch.float32).unsqueeze(0)

            output = model(pixel_values=pixel_values, metadata=metadata)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()

            true_labels.append(sample["label"])
            predicted_labels.append(predicted_label)

    # Oblicz metryki
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))


# if __name__ == "__main__":
#     # main()
#     eval_model()
