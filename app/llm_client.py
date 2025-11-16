from openai import OpenAI
from app.config import OPENAI_API_KEY

def get_llm_client():
    api_key = OPENAI_API_KEY
    return OpenAI(api_key=api_key)

def ask_llm(client, prompt: str, model: str="gpt-5.1"):
    response = client.chat.completions.create(
        model=model,
        message=[
            {"role": "system", "content": "You are an accurate document QA assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    