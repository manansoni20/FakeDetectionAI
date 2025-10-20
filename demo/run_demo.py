# demo/run_demo.py
import sys
import os
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encoders import TextEncoder, ImageEncoder
from src.fusion_model import FusionClassifier
from src.preprocess import load_data

print("\n🧠 Running Fake Detection AI Demo...\n")

# Initialize models
text_encoder = TextEncoder()
image_encoder = ImageEncoder()
fusion_model = FusionClassifier()

# Load CSV data
csv_path = "data/SamplePosts.csv"
df = load_data(csv_path)

labels = ["Fake", "Real"]

# Loop through each post in CSV
for index, row in df.iterrows():
    text = [row['text']]
    image_path = row['image_path']

    # Encode text and image
    text_emb = text_encoder.encode(text)
    image_emb = image_encoder.encode(image_path)

    # Get model prediction
    logits = fusion_model(text_emb, image_emb)
    probs = F.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1)

    # Print result
    print(f"Text: {text[0]}")
    print(f"Image Path: {image_path}")
    print(f"Prediction: {labels[prediction]} (confidence: {probs[0, prediction].item():.2f})\n")

