import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=1000, hidden_dim=512, num_classes=2):
        super(FusionClassifier, self).__init__()
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.image_fc = nn.Linear(image_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_features, image_features):
        text_out = F.relu(self.text_fc(text_features))
        image_out = F.relu(self.image_fc(image_features))
        fused = (text_out + image_out) / 2
        output = self.classifier(fused)
        return output

