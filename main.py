import os
import glob
from dotenv import load_dotenv
from agents.analyzer_agent import ImageAnalyzerAgent
from agents.explainer_agent import ExplainerAgent
from agents.flagger_agent import FlaggerAgent
from utils.pdf_report import PDFReport
from utils.gradcam import generate_gradcam

# Load OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is not set in the .env file.")

def find_image():
    # Search inside nested images folders in ChestX-ray14
    image_candidates = glob.glob(os.path.join("data", "chest_xray14", "images_*", "images", "*.png"))
    if not image_candidates:
        raise FileNotFoundError("❌ No PNG images found inside nested images folders.")
    image_candidates.sort()  # Optional: pick consistent image
    print(f"Found {len(image_candidates)} images")
    return image_candidates[5]  # Pick the 6th image (to avoid blur)

def main():
    print("🧠 Launching Medical Imaging QA Agent...\n")

    image_path = find_image()
    print(f"🖼️ Using image: {image_path}\n")

    analyzer = ImageAnalyzerAgent()
    explainer = ExplainerAgent(api_key)
    flagger = FlaggerAgent()

    if flagger.run(image_path):
        print("⚠️ Image is blurry or unreadable. QA failed.")
    else:
        findings = analyzer.run(image_path)
        print("📋 Detected Abnormalities:")
        print(findings if findings else "✅ No major abnormalities detected.")

        explanation = explainer.run(findings)
        print("\n💬 Explanation:")
        print(explanation)

        # Generate Grad-CAM visualization
        heatmap_path = generate_gradcam(image_path, analyzer.model.model)


        # Generate PDF Report
        report = PDFReport()
        report.generate(
            filename=os.path.basename(image_path),
            findings=findings,
            explanation=explanation,
            gradcam_path=heatmap_path,
            output_path="report_" + os.path.basename(image_path).replace('.png', '.pdf')
        )

if __name__ == "__main__":
    main()
