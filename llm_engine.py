import openai
import os

def query_phi3(prompt):
    client = openai.OpenAI(api_key=os.getenv("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"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
