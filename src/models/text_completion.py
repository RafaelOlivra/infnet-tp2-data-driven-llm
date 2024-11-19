from pydantic import BaseModel


class AutoCompleteModel(BaseModel):
    phrase: str

class AutoCompleteResponseModel(BaseModel):
    response: str