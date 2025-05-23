import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.models import densenet121
import os

class GradCAM:
    def __init__(self, model, target_layer="features.denseblock4"):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.target_layer = dict([*model.named_modules()])[target_layer]
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, index=None):
        input_tensor = input_tensor.to(self.device)

        output = self.model(input_tensor)
        if index is None:
            index = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, index]
        class_score.backward()

        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        weights = gradients.mean(dim=(1, 2))  # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= cam.max()
        return cam

    def save_heatmap(self, cam, original_path, output_path="heatmap.png"):
        original = Image.open(original_path).convert("RGB")
        original = original.resize((224, 224))
        original_np = np.array(original)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlay = heatmap + np.float32(original_np) / 255
        overlay = overlay / np.max(overlay)

        output_img = np.uint8(255 * overlay)
        cv2.imwrite(output_path, output_img)
        print(f"🖼️ Grad-CAM saved to {output_path}")


def generate_gradcam(image_path, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model or densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Linear(1024, 14)
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    gradcam = GradCAM(model, target_layer="features.denseblock4")
    cam = gradcam.generate(input_tensor)
    output_path = "heatmap_" + os.path.basename(image_path)
    gradcam.save_heatmap(cam, image_path, output_path=output_path)
    return output_path
