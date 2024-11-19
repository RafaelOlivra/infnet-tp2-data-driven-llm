from pydantic import BaseModel


class TranslationModel(BaseModel):
    text: str


class TranslationResponseModel(BaseModel):
    translation: str
