import cv2

class QualityChecker:
    def check_blur(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return True
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var < 100
