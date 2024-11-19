import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities import (
    OpenWeatherMapAPIWrapper,
    GoogleSerperAPIWrapper,
)
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.title("Smart Day Planner")

# Initialize external tools
search = GoogleSerperAPIWrapper(
    serper_api_key=os.getenv("SERPER_API_KEY"), gl="br", hl="pt-br", k=5
)
weather = OpenWeatherMapAPIWrapper(
    openweathermap_api_key=os.getenv("OPENWEATHER_API_KEY")
)

tools = [
    Tool(name="Search", func=search.run, description="Search the web for data."),
    Tool(
        name="Weather",
        func=weather.run,
        description="Get current weather in a location.",
    ),
]

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.8,  # Lowered temperature for consistency
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

prompt_template = """
You are a friendly modern day planner who can help users find activities based on preferences and weather.
You have access to the following tools: 
{tools}

Use the following format:

Question: the input question you must answer
Thought: think about what to do and if you need to use tools
Action: use one of [{tool_names}] if needed.
Action Input: the input for the tool
Observation: the result of the action
...
Thought: I now know what to answer
Final Answer: your response to the question

Begin!

Chat History: {history}
Latest Question: {input}

{agent_scratchpad}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["input", "history", "agent_scratchpad", "tools", "tool_names"],
)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)


def initialize_agent_executor():
    """Initialize the AgentExecutor and store it in st.session_state."""
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


def clear_chat_history():
    """Clear the chat history and reinitialize the AgentExecutor."""
    st.session_state.agent_executor.memory.chat_memory.clear()
    initialize_agent_executor()


if "agent_executor" not in st.session_state:
    initialize_agent_executor()

# User query input
query = st.text_input("Ask me anything")

# Execute the query if provided
if query:
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.agent_executor.invoke({"input": query})
            st.info(response["output"], icon="🤖")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display chat history
with st.expander("Chat History"):
    for message in st.session_state.agent_executor.memory.chat_memory.messages:
        st.write(f"{message.type}: {message.content}")

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    clear_chat_history()
