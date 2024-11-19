from fastapi import APIRouter, HTTPException
from langchain_community.llms import HuggingFacePipeline
from transformers import MarianMTModel, MarianTokenizer, pipeline

from ..models.text_translation import TranslationModel, TranslationResponseModel

router = APIRouter()

# Configure the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Set the pipeline for translation
translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)

# Initialize the LangChain pipeline
llm_chain = HuggingFacePipeline(pipeline=translation_pipeline)


def generate_translation(text: str) -> str:
    response = llm_chain(text)
    return response.strip()


@router.post("/english-to-german/", response_model=TranslationResponseModel)
async def english_to_german(body: TranslationModel) -> TranslationResponseModel:
    try:
        translated_text = generate_translation(body.text)
        return TranslationResponseModel(translation=translated_text)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
