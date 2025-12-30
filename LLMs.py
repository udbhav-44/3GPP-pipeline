"""
This module sets up various configurations for different ChatOpenAI models using the langchain library.
"""
import os
from dotenv import load_dotenv

load_dotenv('.env') 
openai_api_key=os.getenv("OPEN_AI_API_KEY_30")
deepseek_api_key=os.getenv("DEEPSEEK_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


def get_llm(model="gpt-4o-mini", temperature=0.4, top_p=0.4, provider="openai"):
    if provider == "deepseek":
        return ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key=deepseek_api_key, 
            openai_api_base='https://api.deepseek.com',
            temperature=temperature,
            top_p=top_p
        )
    else:
         return ChatOpenAI(model=model, openai_api_key=openai_api_key, temperature=temperature, top_p=top_p)

# Keep these for backward compatibility during refactor, but they should eventually be replaced or use default
# Initialize with default to avoid breaking imports immediately, but code should switch to using get_llm
GPT4o_mini_LATS = get_llm(temperature=0.4, top_p=0.4)
GPT4o_mini_GraphGen = get_llm(temperature=0.2, top_p=0.1)
GPT4o_mini_Complex = get_llm(temperature=0.6, top_p=0.7)


_message_histories = {}

def _get_message_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _message_histories:
        _message_histories[session_id] = ChatMessageHistory()
    return _message_histories[session_id]

_conversation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# This global usage is problematic for per-request model switching. 
# Ideally, message history and runnable should be created per request or context.
# For now, we keep it but it might use the default (OpenAI).
conversation_complex = RunnableWithMessageHistory(
    _conversation_prompt | GPT4o_mini_Complex,
    _get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)

def run_conversation_complex(prompt: str, session_id: str = "default") -> str:
    response = conversation_complex.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content if hasattr(response, "content") else str(response)

#TopicalGuardrails
GPT4o_mini_GuardRails = get_llm(temperature=0.2, top_p=0.1)

def reload_llms():
    global openai_api_key
    global deepseek_api_key
    global GPT4o_mini_LATS
    global GPT4o_mini_GraphGen
    global GPT4o_mini_Complex
    global GPT4o_mini_GuardRails
    global conversation_complex

    load_dotenv('.env', override=True)
    openai_api_key = os.getenv("OPEN_AI_API_KEY_30")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    
    GPT4o_mini_LATS = get_llm(temperature=0.4, top_p=0.4)
    GPT4o_mini_GraphGen = get_llm(temperature=0.2, top_p=0.1)
    GPT4o_mini_Complex = get_llm(temperature=0.6, top_p=0.7)
    GPT4o_mini_GuardRails = get_llm(temperature=0.2, top_p=0.1)
    
    conversation_complex = RunnableWithMessageHistory(
        _conversation_prompt | GPT4o_mini_Complex,
        _get_message_history,
        input_messages_key="input",
        history_messages_key="history",
    )

