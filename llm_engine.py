# llm_engine.py
import os
import openai

def query_phi3(prompt: str) -> str:
    """
    Query OpenAI API (gpt-4o-mini) for AI policy interpretation.
    Works on Streamlit Cloud. Returns a safe message if key is missing.
    """
    api_key = os.getenv("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")
    if not api_key:
        return "❌ OPENAI_API_KEY is missing. Please add it in Streamlit Cloud Secrets."

    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ OpenAI API Error: {str(e)}"
