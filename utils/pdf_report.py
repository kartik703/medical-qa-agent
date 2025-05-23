from fpdf import FPDF
from datetime import datetime
from PIL import Image
import os

class PDFReport(FPDF):
    def header(self):
        # Logo (optional)
        if os.path.exists("logo.png"):
            self.image("logo.png", 10, 8, 33)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Medical Imaging QA Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")

    def add_scan_info(self, filename):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Scan Information", ln=True)
        self.set_font("Helvetica", "", 11)
        self.cell(0, 10, f"Filename: {filename}", ln=True)
        self.ln(5)

    def add_findings_table(self, findings):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Detected Abnormalities", ln=True)
        self.set_font("Helvetica", "", 11)
        col_width = 90
        self.cell(col_width, 8, "Condition", 1)
        self.cell(col_width, 8, "Confidence", 1, ln=True)
        for condition, score in findings.items():
            self.cell(col_width, 8, condition, 1)
            self.cell(col_width, 8, str(round(float(score), 3)), 1, ln=True)
        self.ln(5)

    def add_explanation(self, explanation):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Explanation", ln=True)
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 8, explanation)
        self.ln(5)

    def add_gradcam(self, gradcam_path):
        if os.path.exists(gradcam_path):
            self.set_font("Helvetica", "B", 12)
            self.cell(0, 10, "Grad-CAM Visualization", ln=True)
            self.image(gradcam_path, x=30, w=150)
            self.ln(5)

    def generate(self, filename, findings, explanation, gradcam_path=None, output_path="report.pdf"):
        self.add_page()
        self.add_scan_info(filename)
        self.add_findings_table(findings)
        self.add_explanation(explanation)
        if gradcam_path:
            self.add_gradcam(gradcam_path)
        self.output(output_path)
        print(f"📄 PDF report saved to: {output_path}")