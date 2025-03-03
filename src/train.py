from typing import Any

import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments
from utils import FilePath

from src.model import MultiModalClassifier


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([torch.tensor(item["pixel_values"]) for item in batch])
    metadata = torch.stack([torch.tensor(item["metadata"]) for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "metadata": metadata, "labels": labels}


def train_model(preprocesed_dataset: DatasetDict) -> None:
    num_metadata_features = len(["dx", "dx_type", "sex", "localization"]) + 1
    num_labels = len(set(preprocesed_dataset["train"]["label"]))

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
        train_dataset=preprocesed_dataset["train"],
        eval_dataset=preprocesed_dataset["validation"],
        data_collator=collate_fn,
    )

    trainer.train()
    torch.save(model.state_dict(), f"{FilePath.model_vit_skin_cancer}/model.pth")
