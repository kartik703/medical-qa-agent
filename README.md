# 🩻 Medical Imaging QA Agent

An AI-powered diagnostic assistant that:
- Detects abnormalities in chest X-rays using a deep learning model
- Generates easy-to-understand explanations via GPT-4
- Visualizes attention with Grad-CAM heatmaps
- Creates downloadable PDF medical reports



## 📂 How to Run Locally
```bash
git clone https://github.com/yourusername/medical-qa-agent.git
cd medical-qa-agent
pip install -r requirements.txt
streamlit run app.py
```

## 🔐 Setup `.env`
Create a `.env` file:
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxx
```

## 🧠 Powered by
- 🤖 GPT-4 (OpenAI)
- 🧠 DenseNet121 (PyTorch)
- 🎨 Grad-CAM
- 🖥️ Streamlit

## 📄 Outputs
- Detected conditions + confidences
- GPT explanation
- Grad-CAM overlay image
- PDF report

---

## 👨‍⚕️ Disclaimer
This project is for **educational/demo purposes only**. Do not use it for real medical diagnosis.

---

MIT License © 2025 Kartik Goswami