from fastapi import APIRouter, HTTPException
from ..models.text_completion import AutoCompleteModel, AutoCompleteResponseModel
from transformers import pipeline

router = APIRouter()


def generate_response(message: str) -> dict:
    generator = pipeline("text-generation", model="gpt2-large")
    return generator(message)


def format_response(response: dict) -> str:
    return "".join(response.split("\n<|assistant|>\n")[1:]).strip()


@router.post("/phrase/")
async def autocomplete(body: AutoCompleteModel) -> AutoCompleteResponseModel:
    try:
        response = generate_response(body.phrase)
        return AutoCompleteResponseModel(response=response[0]["generated_text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
