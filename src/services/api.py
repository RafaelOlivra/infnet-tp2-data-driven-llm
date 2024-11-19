from fastapi import FastAPI
from ..routers.text_completion import router as text_completion_router
from ..routers.text_translation import router as text_translation_router
from ..routers.text_translation_langchain_gemini import (
    router as text_translation_gemini_router,
)
from ..routers.text_translation_langchain_huggingface import (
    router as text_translation_huggingface_router,
)
from ..routers.chat_fakellm import router as chat_fakellm_router

app = FastAPI()

app.include_router(text_completion_router, prefix="/text-completion")
app.include_router(text_translation_router, prefix="/text-translation")
app.include_router(chat_fakellm_router, prefix="/chat-fakellm")

app.include_router(text_translation_gemini_router, prefix="/gemini/translate")
app.include_router(text_translation_huggingface_router, prefix="/huggingface/translate")


@app.get("/")
async def root():
    return {"message": "Welcome to our AI Powered API!"}
