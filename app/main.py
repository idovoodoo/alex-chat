from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os


app = FastAPI()

# Enable CORS (allows browser/mobile access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# OpenAI v1 client using API key from environment (Render sets this)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatIn(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
def chat(data: ChatIn):
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Reply briefly."},
                {"role": "user", "content": data.message},
            ],
        )

        # response structure may be attribute-accessible or dict-like; handle both
        choice = response.choices[0] if hasattr(response, "choices") else response["choices"][0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            reply = choice.message.content
        else:
            reply = choice["message"]["content"]

        return {"reply": reply}
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from openai import OpenAI
    import os


    app = FastAPI()

    # Enable CORS (allows browser/mobile access)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # OpenAI v1 client using API key from environment (Render sets this)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    class ChatIn(BaseModel):
        message: str


    @app.get("/")
    def root():
        return {"status": "ok"}


    @app.post("/chat")
    def chat(data: ChatIn):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "Reply briefly."},
                    {"role": "user", "content": data.message},
                ],
            )

            # response structure may be attribute-accessible or dict-like; handle both
            choice = response.choices[0] if hasattr(response, "choices") else response["choices"][0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                reply = choice.message.content
            else:
                reply = choice["message"]["content"]

            return {"reply": reply}
        except Exception as e:
            return {"error": str(e)}
