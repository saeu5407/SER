import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
        self.model.classifier = nn.Identity()
        self.classifier = nn.Linear(256, 6)

    def forward(self, x):
        output = self.model(x)
        output = self.classifier(output.logits)
        return output

