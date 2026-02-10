from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatIn(BaseModel):
    message: str


@app.post("/chat")
def chat(data: ChatIn):

    resp = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": data.message}
        ]
    )

    reply = resp["choices"][0]["message"]["content"]

    return {"reply": reply}
