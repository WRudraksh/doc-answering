from openai import OpenAI
from app.config import GROQ_API_KEY

def get_llm_client():
    api_key = GROQ_API_KEY
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

def ask_llm(client, prompt: str, model: str="llama-3.3-70b-versatile"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an accurate document QA assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
    