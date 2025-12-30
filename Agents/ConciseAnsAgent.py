import os
import logging
from dotenv import load_dotenv
from datetime import datetime

from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    PostgresSaver = None
try:
    from langgraph.graph.message import add_messages
except ImportError:
    from langgraph.graph import add_messages


#TO CHANGE IF POSSIBLE
from LLMs import GPT4o_mini_LATS

load_dotenv('.env')
logger = logging.getLogger(__name__)

def _get_checkpoint_conn_str():
    url = os.getenv("LANGGRAPH_CHECKPOINT_URL") or os.getenv("CHECKPOINT_DATABASE_URL")
    if url:
        return url
    user = os.getenv("POSTGRES_USER", "udbhav")
    password = os.getenv("POSTGRES_PASSWORD", "login123")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def _build_checkpointer():
    if PostgresSaver is None:
        return MemorySaver()
    conn_str = _get_checkpoint_conn_str()
    try:
        saver = PostgresSaver.from_conn_string(conn_str)
        if hasattr(saver, "setup"):
            saver.setup()
        return saver
    except Exception as exc:
        logger.warning("Postgres checkpointer unavailable (%s); falling back to memory.", exc)
        return MemorySaver()

CHECKPOINTER = _build_checkpointer()

class MemoryState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def _build_memory_graph():
    builder = StateGraph(MemoryState)
    builder.add_node("noop", lambda state: state)
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    return builder.compile(checkpointer=CHECKPOINTER)

MEMORY_GRAPH = _build_memory_graph()
MEMORY_NODE = "noop"

def _get_thread_config(thread_id):
    if not thread_id:
        return None
    thread_key = f"concise:{thread_id}"
    return {
        "configurable": {
            "thread_id": str(thread_key),
        }
    }


def conciseAns_vanilla(query, tools_list, thread_id=None):
    logger.info("Running conciseAns_vanilla")
    system_prompt = f"""
        Note: The Current Date and Time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. All your searches and responses
        must be with respect to this time frame. Whenever an agent is called, use this date time as the current date time
        if required.

        Provide a concise, and well researched answer for the following query.
        Provide Numbers, Facts, Concepts, Formulaes, Research Insights in order to back the answer.

        Give a direct answer to the question, without showing your thought process,
        extra information, and don't provide detailed analysis. Only answer what is asked
        with the NECESSARY information backing it.
        If the answer is already in the chat history, use it and do not call tools.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(GPT4o_mini_LATS, tools_list, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true",
    )
    payload = {"input": query}
    config = _get_thread_config(thread_id)
    if config:
        try:
            snapshot = MEMORY_GRAPH.get_state(config)
            if hasattr(snapshot, "values") and isinstance(snapshot.values, dict):
                history_messages = snapshot.values.get("messages") or []
                if history_messages:
                    payload["chat_history"] = history_messages
            elif isinstance(snapshot, dict):
                history_messages = snapshot.get("messages") or []
                if history_messages:
                    payload["chat_history"] = history_messages
        except Exception as exc:
            logger.exception("Concise memory read failed")
    response = agent_executor.invoke(payload)
    if config:
        try:
            MEMORY_GRAPH.update_state(
                config,
                {"messages": [HumanMessage(content=query), AIMessage(content=response["output"])]},
                as_node=MEMORY_NODE,
            )
        except Exception as exc:
            logger.exception("Concise memory update failed")
    with open("conciseResponse.md", "w") as f:
        f.write(response['output'])
    logger.info("Completed conciseAns_vanilla")
    logger.debug("Concise response type: %s", type(response))
    return response['output']



