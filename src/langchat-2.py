import streamlit as st
from langchain_core.prompts import MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain  # Add this import
from langchain.prompts import PromptTemplate  # For wrapping the prompt in LLMChain
from langchain.agents import AgentExecutor, Tool, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities import (
    OpenWeatherMapAPIWrapper,
    GoogleSerperAPIWrapper,
)
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.title("Smart Day Planner")

# API keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize external tools
search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY, hl="pt", k=5)
weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=OPENWEATHER_API_KEY)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to search the web for data.",
    ),
    Tool(
        name="Weather",
        func=weather.run,
        description="Useful for when you need to get the current weather in a location.",
    ),
]

# System message for the agent
prefix = """ You are a friendly modern day planner.
You can help users to find activities in a given city based
on their preferences and the weather.
You have access to the two tools:
"""

suffix = """
Chat History:
{chat_history}
Latest Question: {input}
{agent_scratchpad}
"""

# Initialize chat message history
msgs = StreamlitChatMessageHistory()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        chat_memory=msgs, memory_key="chat_history", return_messages=True
    )

memory = st.session_state.memory

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.8,
    google_api_key=GOOGLE_API_KEY,
)

# Create the agent prompt
prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

# Wrap LLM and prompt in an LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Create the agent
agent = ConversationalAgent(llm_chain=llm_chain, tools=tools, prompt=prompt)

# Create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
)

# User query input
query = st.text_input("Ask me anything")

# Execute the query if provided
if query:
    with st.spinner("Thinking..."):
        try:
            response = agent_executor.run({"input": query})
            print(response)
            st.info(response, icon="🤖")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display chat history
with st.expander("Chat History"):
    for message in st.session_state.memory.chat_memory.messages:
        st.write(f"{message.type}: {message.content}")
