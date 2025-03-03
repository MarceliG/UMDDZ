import torch
from torch import nn
from transformers import ViTModel


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
