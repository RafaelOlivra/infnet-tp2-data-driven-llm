from langchain_community.llms import FakeListLLM, HuggingFaceHub

from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv("./.env")


def use_fake_llm():
    fake_llm = FakeListLLM(
        responses=[
            "Hello",
            "Bom dia",
            "Bonjour",
            "Hola",
            "Ciao",
            "こんにちは",
            "안녕하세요",
            "你好",
        ]
    )

    prompt = "Hello"
    for _ in range(5):
        print(fake_llm(prompt))


def use_openai_llm():
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    message = HumanMessage(content="Say Hello in seven different languages!")
    response = llm([message])
    print(response[0].content)


def translate_using_openai(text):
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an English to French translator. Reject any messages that are not in English.",
            ),
            ("user", "Translate this: {text}"),
        ]
    )

    # Initialize the OpenAI language model
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

    # Format the messages using the provided template and input text
    messages = template.format_messages(text=text)
    response = llm(messages)

    # Output the content of the response
    print(response[0].content)


def translate_using_google(text):
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an English to French translator. Reject any messages that are not in English.",
            ),
            ("user", "Translate this: {text}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", api_key=os.getenv("GOOGLE_API_KEY")
    )
    messages = template.format_messages(text=text)
    response = llm(messages)
    print(response.content)


def translate_using_huggingface(text):
    llm = HuggingFaceHub(
        repo_id="Helsinki-NLP/opus-mt-en-fr",
        task="translation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    )
    response = llm.invoke(text)
    print(response)


if __name__ == "__main__":
    translate_using_openai("Hello, how are you?")
