from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18

class TextEncoder:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

class ImageEncoder:
    def __init__(self):
        self.model = resnet18(weights='IMAGENET1K_V1')
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def encode(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            embeddings = self.model(img_tensor)
        return embeddings

