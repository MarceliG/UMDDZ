import os
import random
from datetime import datetime

import torch
from datasets import DatasetDict
from torchvision.transforms.functional import to_pil_image
from utils import FilePath, logger

from src.model import MultiModalClassifier

label2name = {
    0: "melanoma",
    1: "nevus",
    2: "basal cell carcinoma",
    3: "seborrheic keratosis",
    4: "vascular lesion",
    5: "dermatofibroma",
    6: "actinic keratosis",
}


def denormalize_image(tensor, mean, std):
    """Odnormalizacja tensora RGB"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)  # Ważne: ograniczamy wartości do [0,1]


def predict_random_sample(preproced_dataset: DatasetDict) -> None:
    os.makedirs(FilePath.predictions, exist_ok=True)

    # Setup model
    num_metadata_features = len(["dx", "dx_type", "sex", "localization"]) + 1
    num_labels = len(set(preproced_dataset["train"]["label"]))

    model = MultiModalClassifier(num_metadata_features, num_labels)
    model.load_state_dict(torch.load(f"{FilePath.model_vit_skin_cancer}/model.pth"))
    model.eval()

    # Pick a random test sample
    sample = random.choice(preproced_dataset["test"])

    pixel_values_tensor = torch.tensor(sample["pixel_values"], dtype=torch.float32)
    metadata_tensor = torch.tensor(sample["metadata"], dtype=torch.float32).unsqueeze(0)
    pixel_values_batch = pixel_values_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(pixel_values=pixel_values_batch, metadata=metadata_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

    true_label = sample["label"]

    true_name = label2name.get(true_label, "Unknown")
    predicted_name = label2name.get(predicted_label, "Unknown")

    # Log to terminal
    logger.info(f"True label: {true_label} ({true_name})")
    logger.info(f"Predicted label: {predicted_label} ({predicted_name})")

    # Prepare folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(FilePath.predictions, f"prediction_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_denorm = denormalize_image(pixel_values_tensor.clone(), mean, std)

    # Save image
    image_path = os.path.join(folder_path, "image.png")
    img = to_pil_image(img_denorm)
    img.save(image_path)

    # Save metadata and prediction
    text_path = os.path.join(folder_path, "prediction.txt")
    with open(text_path, "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"True label: {true_label} ({true_name})\n")
        f.write(f"Predicted label: {predicted_label} ({predicted_name})\n")
        f.write(f"Metadata: {sample['metadata']}\n")
        f.write(f"Probabilities: {probabilities.squeeze().tolist()}\n")
        f.write("Image file: image.png\n")

    logger.info(f"Prediction saved to {text_path}")
    logger.info(f"Image saved to {image_path}")
