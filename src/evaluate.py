import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import FilePath, logger

from src.model import MultiModalClassifier


def evaluate_model(preproced_dataset: DatasetDict) -> None:
    num_metadata_features = len(["dx", "dx_type", "sex", "localization"]) + 1
    num_labels = len(set(preproced_dataset["train"]["label"]))

    model = MultiModalClassifier(num_metadata_features, num_labels)
    model.load_state_dict(torch.load(f"{FilePath.model_vit_skin_cancer}/model.pth"))
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for sample in preproced_dataset["test"]:
            pixel_values = torch.tensor(sample["pixel_values"], dtype=torch.float32).unsqueeze(0)
            metadata = torch.tensor(sample["metadata"], dtype=torch.float32).unsqueeze(0)

            output = model(pixel_values=pixel_values, metadata=metadata)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()

            true_labels.append(sample["label"])
            predicted_labels.append(predicted_label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    logger.info(f"Accuracy: {accuracy:.4f}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(true_labels, predicted_labels, digits=4))

    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(true_labels, predicted_labels))
