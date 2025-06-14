import os
import io
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

class ModelService:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance

    def __init__(self, checkpoint_path: str = None):
        if hasattr(self, 'model'):
            return

        if checkpoint_path is None:
            checkpoint_path = os.environ.get('MODEL_PATH', 'models/best_model.pth')
        
        print("Initializing Model Service...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        p_checkpoint = Path(checkpoint_path)
        if not p_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {p_checkpoint}")
            
        checkpoint = torch.load(p_checkpoint, map_location=self.device, weights_only=False)
        
        self.tag_vocab = checkpoint['tag_vocab']
        num_classes = len(self.tag_vocab)
        print(f"Number of classes from checkpoint: {num_classes}")
        
        train_args = checkpoint.get('args', {})
        image_size = train_args.get('image_size', 384)
        print(f"Image size from checkpoint: {image_size}x{image_size}")

        KNOWN_MODEL_NAME = "tf_efficientnetv2_s.in21k"
        
        print(f"Creating empty model with known architecture: {KNOWN_MODEL_NAME}")
        self.model = timm.create_model(
            model_name=KNOWN_MODEL_NAME,
            pretrained=False,
            num_classes=num_classes
        )

        weights = checkpoint['model_state_dict']
        self.model.load_state_dict(weights)
        print("Weights loaded successfully into the model.")
        
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("✅ Model Service is ready.")

    def predict_tags(self, image_bytes: bytes, top_n: int = 15) -> List[str]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0)

        _, top_indices = torch.topk(probabilities, top_n)
        top_tags = [self.tag_vocab[i] for i in top_indices.cpu().tolist()]
        
        return top_tags

# Создаем экземпляр сервиса для импорта в main.py
model_service = ModelService()
