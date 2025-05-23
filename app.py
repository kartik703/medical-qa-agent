import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from agents.analyzer_agent import ImageAnalyzerAgent
from agents.explainer_agent import ExplainerAgent
from agents.flagger_agent import FlaggerAgent
from utils.gradcam import generate_gradcam
from utils.pdf_report import PDFReport

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="🩻 Medical Imaging QA", layout="wide")
st.title("🩻 Medical Imaging QA Agent")

st.markdown("Upload a chest X-ray image to detect abnormalities, see visual explanations, and download a medical PDF report.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with open("temp_input.png", "wb") as f:
        f.write(uploaded_file.read())

    st.image("temp_input.png", caption="Uploaded X-ray", width=300)

    st.markdown("---")
    with st.spinner("Analyzing image..."):
        # Run agents
        analyzer = ImageAnalyzerAgent()
        explainer = ExplainerAgent(api_key)
        flagger = FlaggerAgent()

        if flagger.run("temp_input.png"):
            st.warning("⚠️ The uploaded image appears blurry or low quality.")
        else:
            findings = analyzer.run("temp_input.png")
            explanation = explainer.run(findings)

            st.subheader("📋 Detected Abnormalities")
            if findings:
                for k, v in findings.items():
                    st.markdown(f"**{k}**: {round(float(v), 3)}")
            else:
                st.success("✅ No major abnormalities detected.")

            st.subheader("💬 GPT Explanation")
            st.markdown(explanation)

            st.subheader("🖼️ Grad-CAM Visualization")
            heatmap_path = generate_gradcam("temp_input.png", analyzer.model.model)
            st.image(heatmap_path, caption="Grad-CAM Overlay", use_column_width=True)

            # Generate PDF
            report_filename = f"report_{os.path.basename(uploaded_file.name).replace('.', '_')}.pdf"
            report = PDFReport()
            report.generate(
                filename=os.path.basename(uploaded_file.name),
                findings=findings,
                explanation=explanation,
                gradcam_path=heatmap_path,
                output_path=report_filename
            )

            with open(report_filename, "rb") as f:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=f,
                    file_name=report_filename,
                    mime="application/pdf"
                )

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, PyTorch, and GPT-4")
