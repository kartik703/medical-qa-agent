from models.quality_checker import QualityChecker

class FlaggerAgent:
    def __init__(self):
        self.qc = QualityChecker()

    def run(self, img_path):
        return self.qc.check_blur(img_path)
