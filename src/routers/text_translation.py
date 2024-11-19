from fastapi import APIRouter, HTTPException
from ..models.text_translation import TranslationModel, TranslationResponseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

router = APIRouter()

# Configure the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_response(text: str) -> str:
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Generate translation using the model
    outputs = model.generate(**inputs, max_length=100, num_beams=4)

    # Decode the output to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


@router.post("/english-to-french/")
async def english_to_french(body: TranslationModel) -> TranslationResponseModel:
    try:
        translated_text = generate_response(body.text)
        return TranslationResponseModel(translation=translated_text)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
