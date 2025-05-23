import openai
from openai import OpenAI

class ExplainerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def run(self, findings):
        if not findings:
            return "No major abnormalities detected."

        condition_list = ', '.join(findings.keys())
        prompt = f"Explain the following chest X-ray findings in simple, patient-friendly language: {condition_list}"

        response = self.client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" if preferred
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
