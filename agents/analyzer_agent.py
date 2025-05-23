from models.vision_model import VisionModel

class ImageAnalyzerAgent:
    def __init__(self):
        self.model = VisionModel()

    def run(self, image_path):
        probs = self.model.predict(image_path)
        labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
        return {label: round(p, 3) for label, p in zip(labels, probs) if p > 0.5}
