from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os


app = FastAPI()

# Enable CORS (allows browser/mobile access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# OpenAI API key from Render environment
openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatIn(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
def chat(data: ChatIn):

    response = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": data.message},
        ],
    )

    reply = response["choices"][0]["message"]["content"]

    return {"reply": reply}
