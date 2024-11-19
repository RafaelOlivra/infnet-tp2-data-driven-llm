from fastapi import APIRouter
from langchain_community.llms.fake import FakeListLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from ..models.chat import ChatModel, ChatResponseModel

router = APIRouter()

# Simulated responses
fake_responses = [
    "Olá! Como posso ajudar?",
    "Sou um chatbot simples. Posso responder perguntas básicas.",
    "Não entendi :(",
    "Obrigado por entrar em contato! Até mais!",
]
fake_llm = FakeListLLM(responses=fake_responses)  # Use llm module

# Chain configuration
llm_chain = LLMChain(
    llm=fake_llm, prompt=ChatPromptTemplate(messages=["user", "assistant"])
)


def generate_response(message: str) -> dict:
    response = llm_chain.run({"user": message})
    return response


@router.post("/chat/")
async def chat(body: ChatModel):
    response = generate_response(body.message)
    return ChatResponseModel(assistant=response)
