import torch
import torchvision.transforms as T
from torchvision.models import densenet121
from PIL import Image

class VisionModel:
    def __init__(self):
        # Load pretrained DenseNet and modify for 14 chest X-ray conditions
        self.model = densenet121(weights='DEFAULT')
        self.model.classifier = torch.nn.Linear(1024, 14)

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(self.device)

        # Image transformation: resize + convert to tensor + normalize
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, img_path):
        # Load and preprocess image as RGB
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)

        # Run model
        with torch.no_grad():
            output = torch.sigmoid(self.model(img))

        return output.cpu().numpy().flatten()
