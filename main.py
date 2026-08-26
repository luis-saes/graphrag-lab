import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")
MAX_OUTPUT_TOKENS = 500

client = genai.Client(api_key=GEMINI_API_KEY)


def complete(prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    resp = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    return resp.text


if __name__ == "__main__":
    try:
        print(complete("Explain what a vector embedding is in two sentences."))
    finally:
        client.close()
