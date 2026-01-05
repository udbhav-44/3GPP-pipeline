"""
This script sets up a WebSocket server to handle various types of queries and tasks using different agents and APIs.
Functions:
    mainBackend(query, websocket, rag):
        Handles the main backend processing of queries, including classification, planning, and execution of tasks using various agents.
    handle_connection(websocket):
        Manages incoming WebSocket connections and routes messages to the appropriate handlers.
    main():
        Starts the WebSocket server and keeps it running indefinitely.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import uuid
import logging
from logging_config import setup_logging

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=False)
setup_logging()
logger = logging.getLogger("backend")

import time
import contextlib
import json
import re
# Import custom agents for different tasks
from Agents.Agents import Agent
from Agents.Smack import Smack
from Agents.ClassifierAgent import classifierAgent, classifierAgent_RAG
from Agents.PlannerAgent import plannerAgent, plannerAgent_rag
from Agents.DrafterAgent import drafterAgent_vanilla, drafterAgent_rag
from Agents.ConciseAnsAgent import conciseAns_vanilla
from Agents.RAG_Agent import ragAgent
from Agents.LATS.OldfinTools import *
import asyncio
import websockets
from langchain.globals import set_verbose
from makeGraphJSON import makeGraphJSON
from GenerateQuestions import  genQuestionSimple

set_verbose(os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true")
now = time.time()
PROCESS_LOG_PATH = BASE_DIR / "ProcessLogs.md"
GRAPH_PATH = BASE_DIR / "Graph.json"
BAD_QUESTION_PATH = BASE_DIR / "Bad_Question.md"
OUTPUT_DIR = BASE_DIR / "output"
WRITE_ARTIFACTS = os.getenv("WRITE_ARTIFACTS", "false").lower() in {"1", "true", "yes"}

def should_abort(cancel_event):
    return cancel_event is not None and cancel_event.is_set()

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

def ensure_checkpoint_tables() -> bool:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed; Postgres checkpointing disabled."
        )
        return False

    conn_str = _get_checkpoint_conn_str()
    try:
        with PostgresSaver.from_conn_string(conn_str) as saver:
            if hasattr(saver, "setup"):
                saver.setup()
        logger.info("Postgres checkpoint tables are ready.")
        return True
    except Exception:
        logger.exception("Failed to initialize Postgres checkpointer.")
        return False

CHECKPOINT_READY = ensure_checkpoint_tables()

def _build_thread_filters(thread_id: str):
    base = str(thread_id)
    exact_ids = {base, f"concise:{base}"}
    prefixes = {f"{base}:%"}
    return exact_ids, prefixes

def delete_thread_state(thread_id: str) -> bool:
    try:
        import psycopg
    except ImportError:
        logger.warning("psycopg not installed; skipping thread delete for %s", thread_id)
        return False
    if not CHECKPOINT_READY:
        logger.info("Checkpoint tables not ready; skipping thread delete for %s", thread_id)
        return False

    conn_str = _get_checkpoint_conn_str()
    exact_ids, prefixes = _build_thread_filters(thread_id)
    tables = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
    deleted_rows = 0

    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT to_regclass(%s), to_regclass(%s), to_regclass(%s)",
                    ("checkpoint_writes", "checkpoint_blobs", "checkpoints"),
                )
                if not all(cur.fetchone() or []):
                    logger.info("Checkpoint tables missing; skipping delete for %s", thread_id)
                    return False
                for table in tables:
                    for tid in exact_ids:
                        cur.execute(
                            f"DELETE FROM {table} WHERE thread_id = %s",
                            (tid,),
                        )
                        deleted_rows += cur.rowcount
                    for prefix in prefixes:
                        cur.execute(
                            f"DELETE FROM {table} WHERE thread_id LIKE %s",
                            (prefix,),
                        )
                        deleted_rows += cur.rowcount
            conn.commit()
        logger.info("Deleted %s checkpoint rows for thread %s", deleted_rows, thread_id)
        return True
    except Exception:
        logger.exception("Failed to delete thread state for %s", thread_id)
        return False

async def mainBackend(query, websocket, rag, model=None, provider=None, allow_web_tools=False, cancel_event=None, thread_id=None, user_id=None):
    """
    Main backend function to process queries and interact with a websocket.
    This function handles different types of queries (simple or complex) and 
    processes them using various agents and pipelines. The function 
    also generates and sends responses back through the websocket.
    Args:
        query (str): The input query to be processed.
        websocket (WebSocket): The websocket connection to send responses.
        rag (bool): Flag to indicate if RAG mode is enabled.
    Returns:
        None
    """

    if should_abort(cancel_event):
        return
    user_token = None
    try:
        user_token = set_current_user_id(user_id)
    except Exception:
        logger.exception("Failed to set user context")
    logger.info("Running mainBackend: %s", query)
    if model and "gpt-5" in str(model).lower():
        logger.warning("GPT-5 is disabled; falling back to default model.")
        model = None
    GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY_30')
    OPENAI_API_KEY = os.getenv('OPEN_AI_API_KEY_30')
    if WRITE_ARTIFACTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    LLM = 'OPENAI'
    key_dict = {
        'OPENAI': OPENAI_API_KEY,
        'GEMINI': GOOGLE_API_KEY
    }
    api_key = key_dict[LLM]

    IS_RAG = rag

    if IS_RAG == True:
        logger.info("RAG is ON")
    else:
        logger.info("RAG is OFF")

    if WRITE_ARTIFACTS:
        with open(PROCESS_LOG_PATH, "w") as f:
            f.write("")

    resp = ''
    additionalQuestions = None
    send_questions = False
    addn_questions = []

    if should_abort(cancel_event):
        return
    web_tool_names = {"web_search", "web_scrape", "web_search_simple"}

    def filter_web_tools(tool_names):
        if allow_web_tools:
            return tool_names
        return [tool for tool in tool_names if tool not in web_tool_names]

    if not IS_RAG:
        logger.info("Running without internal docs context")
        if should_abort(cancel_event):
            return
        raw_query_type = classifierAgent(query, model=model, provider=provider)
        query_type = raw_query_type.lower().strip()
        if "simple" in query_type:
            query_type = "simple"
        elif "complex" in query_type:
            query_type = "complex"
        else:
            logger.warning("Unexpected classifier output '%s'; defaulting to simple.", raw_query_type)
            query_type = "simple"
        if query_type == "complex":
            logger.info("Running complex task pipeline")
            if should_abort(cancel_event):
                return
            
            # plan -> dict
            plan = plannerAgent(query, model=model, provider=provider, allow_web_tools=allow_web_tools)
            if should_abort(cancel_event):
                return
            #This is the dictionary for UI Graph Construction
            dic_for_UI_graph = makeGraphJSON(plan['sub_tasks'])
            logger.debug("Graph payload: %s", dic_for_UI_graph)
            await asyncio.sleep(1)
            if should_abort(cancel_event):
                return
            await websocket.send(json.dumps({"type": "graph", "response": json.dumps(dic_for_UI_graph)}))
            if WRITE_ARTIFACTS:
                with open(GRAPH_PATH, 'w') as fp:
                    json.dump(dic_for_UI_graph, fp)
            
            out_str = ''''''
            agentsList = []
            
            for sub_task in plan['sub_tasks']:
                if should_abort(cancel_event):
                    return
                addn_questions.append(plan['sub_tasks'][sub_task]['content'])
                agent_name = plan['sub_tasks'][sub_task]['agent']
                agent_role = plan['sub_tasks'][sub_task]['agent_role_description']
                local_constraints = plan['sub_tasks'][sub_task]['local_constraints']
                task = plan['sub_tasks'][sub_task]['content']
                dependencies = plan['sub_tasks'][sub_task]['require_data']
                tools_list = filter_web_tools(plan['sub_tasks'][sub_task]['tools'])
                agent_state = 'vanilla'
                logger.info("Processing agent: %s", agent_name)
                agent = Agent(
                    sub_task,
                    agent_name,
                    agent_role,
                    local_constraints,
                    task,
                    dependencies,
                    tools_list,
                    agent_state,
                    thread_id=thread_id,
                    model=model,
                    provider=provider,
                    allow_web_tools=allow_web_tools,
                )
                agentsList.append(agent)
            
            # Execute the task results using the Smack agent
            smack = Smack(agentsList)
            taskResultsDict = smack.executeSmack()
            if should_abort(cancel_event):
                return
            for task in taskResultsDict:
                out_str += f'{taskResultsDict[task]} \n'
            resp = drafterAgent_vanilla(query, out_str, model=model, provider=provider)
            if should_abort(cancel_event):
                return
            resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
            # resp = generate_chart(resp)
            additionalQuestions = await genQuestionSimple(addn_questions, model=model, provider=provider)
            send_questions = True
            if should_abort(cancel_event):
                return


        elif query_type == "simple":
            logger.info("Running simple task pipeline")
            async def executeSimplePipeline(query):
                tools_list = [web_search_simple] if allow_web_tools else []
                resp = conciseAns_vanilla(query, tools_list, thread_id=thread_id, model=model, provider=provider)   
                resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
                return str(resp)

            resp = await executeSimplePipeline(query)
            additionalQuestions = []
            if should_abort(cancel_event):
                return

    elif IS_RAG:
        logger.info("Running internal docs RAG")
        rag_context = ragAgent(query, state="concise", model=model, provider=provider)
        if should_abort(cancel_event):
            return
        raw_query_type = classifierAgent_RAG(query, rag_context, model=model, provider=provider)
        query_type = raw_query_type.lower().strip()
        if "simple" in query_type:
            query_type = "simple"
        elif "complex" in query_type:
            query_type = "complex"
        else:
            logger.warning("Unexpected RAG classifier output '%s'; defaulting to simple.", raw_query_type)
            query_type = "simple"
        logger.info("RAG query type: %s", query_type)
        
        if query_type == "complex":
            agent_state = 'RAG'
            logger.info("Running complex task pipeline")

            rag_context = ragAgent(query, state="report", model=model, provider=provider)
            if should_abort(cancel_event):
                return
            plan = plannerAgent_rag(query, rag_context, model=model, provider=provider, allow_web_tools=allow_web_tools)
            if should_abort(cancel_event):
                return
            
            dic_for_UI_graph = makeGraphJSON(plan['sub_tasks'])
            for node in dic_for_UI_graph['nodes']:
                node['metadata']['tools'].append('retrieve_documents')

            logger.debug("Graph payload: %s", dic_for_UI_graph)
            await asyncio.sleep(1)
            if should_abort(cancel_event):
                return
            await websocket.send(json.dumps({"type": "graph", "response": json.dumps(dic_for_UI_graph)}))
            if WRITE_ARTIFACTS:
                with open(GRAPH_PATH, 'w') as fp:
                    json.dump(dic_for_UI_graph, fp)
            
            out_str = ''''''
            agentsList = []
            
            for sub_task in plan['sub_tasks']:
                if should_abort(cancel_event):
                    return
                addn_questions.append(plan['sub_tasks'][sub_task]['content'])
                agent_name = plan['sub_tasks'][sub_task]['agent']
                agent_role = plan['sub_tasks'][sub_task]['agent_role_description']
                local_constraints = plan['sub_tasks'][sub_task]['local_constraints']
                task = plan['sub_tasks'][sub_task]['content']
                dependencies = plan['sub_tasks'][sub_task]['require_data']
                tools_list = filter_web_tools(plan['sub_tasks'][sub_task]['tools'])
                logger.info("Processing agent: %s", agent_name)
                agent = Agent(
                    sub_task,
                    agent_name,
                    agent_role,
                    local_constraints,
                    task,
                    dependencies,
                    tools_list,
                    agent_state,
                    thread_id=thread_id,
                    model=model,
                    provider=provider,
                    allow_web_tools=allow_web_tools,
                )
                agentsList.append(agent)
            
            # Execute the task results using the Smack agent
            smack = Smack(agentsList)
            taskResultsDict = smack.executeSmack()
            if should_abort(cancel_event):
                return
            for task in taskResultsDict:
                out_str += f'{taskResultsDict[task]} \n'
            resp = drafterAgent_rag(query, rag_context, out_str, model=model, provider=provider)
            if should_abort(cancel_event):
                return
            resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
            # resp = generate_chart(resp)
            additionalQuestions = await genQuestionSimple(addn_questions, model=model, provider=provider)
            send_questions = True
            if should_abort(cancel_event):
                return

                
            if WRITE_ARTIFACTS:
                with open(OUTPUT_DIR / 'drafted_response.md', 'w') as f:
                    f.write(str(resp))

        elif query_type == 'simple':
            logger.info("Running simple task pipeline")
            resp = rag_context
            resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
            additionalQuestions = []
            if should_abort(cancel_event):
                return
                
    await asyncio.sleep(1)
    if should_abort(cancel_event):
        return
    await websocket.send(json.dumps({"type": "response", "response": resp}))
    if send_questions:
        if additionalQuestions is None:
            additionalQuestions = []
        elif not isinstance(additionalQuestions, list):
            try:
                additionalQuestions = list(additionalQuestions)
            except TypeError:
                additionalQuestions = [str(additionalQuestions)]
        logger.debug("Additional questions: %s", additionalQuestions)
        await asyncio.sleep(1)
        if should_abort(cancel_event):
            return
        await websocket.send(json.dumps({"type": "questions", "response": additionalQuestions[:3]}))
    if user_token is not None:
        try:
            reset_current_user_id(user_token)
        except Exception:
            logger.exception("Failed to reset user context")


async def handle_connection(websocket):
    """
    Manages incoming WebSocket connections and routes messages to the appropriate handlers.
    
    Args:
        websocket (WebSocket): The WebSocket connection to manage.
    
    Returns:
        None: The function processes messages asynchronously and sends back responses to the client.
    """
    rag = False
    allow_web_tools = False
    active_task = None
    active_cancel_event = None
    active_response_id = None
    connection_thread_id = f"conn-{uuid.uuid4().hex}"
    connection_user_id = None

    async def start_query(data):
        nonlocal active_task, active_cancel_event, active_response_id
        nonlocal connection_thread_id, connection_user_id
        if active_task and not active_task.done():
            if active_cancel_event:
                active_cancel_event.set()
            active_task.cancel()

        thread_id = data.get("thread_id")
        if thread_id:
            connection_thread_id = str(thread_id)
        else:
            thread_id = connection_thread_id

        user_payload = data.get("user")
        user_id = data.get("user_id")
        if not user_id and isinstance(user_payload, dict):
            user_id = user_payload.get("id") or user_payload.get("email")
        if user_id:
            connection_user_id = str(user_id)
        else:
            user_id = connection_user_id

        active_cancel_event = asyncio.Event()
        active_response_id = data.get("response_id")

        async def runner():
            try:
                web_tools_flag = allow_web_tools
                if "web_tools" in data:
                    web_tools_flag = bool(data.get("web_tools"))
                await mainBackend(
                    data['query'],
                    websocket,
                    rag,
                    model=data.get("model"),
                    provider=data.get("provider"),
                    allow_web_tools=web_tools_flag,
                    cancel_event=active_cancel_event,
                    thread_id=thread_id,
                    user_id=user_id,
                )
            except asyncio.CancelledError:
                return

        active_task = asyncio.create_task(runner())

    try:
        async for message in websocket:
            data = json.loads(message)
            if data['type'] == 'query':
                logger.info("Received query: %s", data['query'])
                await start_query(data)

            if data['type'] == 'abort':
                if active_task and not active_task.done():
                    if active_response_id and data.get("response_id") and data.get("response_id") != active_response_id:
                        continue
                    if active_cancel_event:
                        active_cancel_event.set()
                    active_task.cancel()

            if data['type'] == 'delete_thread':
                target_thread_id = data.get("thread_id") or connection_thread_id
                logger.info("Deleting thread: %s", target_thread_id)
                if active_task and not active_task.done():
                    if active_cancel_event:
                        active_cancel_event.set()
                    active_task.cancel()
                success = delete_thread_state(str(target_thread_id))
                if target_thread_id == connection_thread_id:
                    connection_thread_id = f"conn-{uuid.uuid4().hex}"
                await websocket.send(
                    json.dumps(
                        {
                            "type": "thread_deleted",
                            "thread_id": str(target_thread_id),
                            "success": success,
                        }
                    )
                )

            if data['type'] == 'toggleRag':
                logger.info("Received toggleRag")
                if "query" in data:
                    rag = bool(data["query"])

            if data['type'] == 'toggleWebTools':
                logger.info("Received toggleWebTools")
                if "query" in data:
                    allow_web_tools = bool(data["query"])
                else:
                    rag = not rag
    except websockets.exceptions.ConnectionClosed as exc:
        logger.info("WebSocket closed: %s", exc)
    except Exception:
        logger.exception("Unexpected error in websocket handler")
    finally:
        if active_task and not active_task.done():
            if active_cancel_event:
                active_cancel_event.set()
            active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active_task


def _get_env_seconds(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"none", "null", "disabled", "false", "0"}:
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, value, default)
        return default

async def main():
    """       
     Starts the WebSocket server and keeps it running indefinitely.
    """
    logger.info("WebSocket server starting on ws://0.0.0.0:8080")
    ping_interval = _get_env_seconds("WS_PING_INTERVAL", 20)
    ping_timeout = _get_env_seconds("WS_PING_TIMEOUT", 20)
    close_timeout = _get_env_seconds("WS_CLOSE_TIMEOUT", 10)
    async with websockets.serve(
        handle_connection,
        "0.0.0.0",
        8080,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
        close_timeout=close_timeout,
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")
