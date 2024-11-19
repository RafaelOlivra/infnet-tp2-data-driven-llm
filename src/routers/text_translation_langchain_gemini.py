import os
from fastapi import APIRouter, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from ..models.text_translation import TranslationModel, TranslationResponseModel

# Load environment variables
load_dotenv()

# Set the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

router = APIRouter()

# Chain configuration
llm_chain = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.8,
    google_api_key=GOOGLE_API_KEY,
)


def generate_response(text: str) -> str:
    messages = [
        (
            "system",
            "Translate the text below from English to French. Return the translated text only.",
        ),
        ("user", text),
    ]

    try:
        response = llm_chain.invoke(messages)
        translated_text = response.content.strip()
        return translated_text
    except Exception as e:
        raise ValueError(f"Error generating response: {str(e)}")


@router.post("/english-to-french/", response_model=TranslationResponseModel)
async def english_to_french(body: TranslationModel) -> TranslationResponseModel:
    try:
        translated_text = generate_response(body.text)
        return TranslationResponseModel(translation=translated_text)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
