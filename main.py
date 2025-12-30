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
from TopicalGuardrails import applyTopicalGuardails
from GenerateQuestions import  genQuestionSimple

set_verbose(os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true")
now = time.time()
PROCESS_LOG_PATH = BASE_DIR / "ProcessLogs.md"
GRAPH_PATH = BASE_DIR / "Graph.json"
BAD_QUESTION_PATH = BASE_DIR / "Bad_Question.md"
OUTPUT_DIR = BASE_DIR / "output"

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

    conn_str = _get_checkpoint_conn_str()
    exact_ids, prefixes = _build_thread_filters(thread_id)
    tables = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
    deleted_rows = 0

    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
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

async def mainBackend(query, websocket, rag, model="gpt-4o-mini", cancel_event=None, thread_id=None):
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
    logger.info("Running mainBackend: %s", query)
    GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY_30')
    OPENAI_API_KEY = os.getenv('OPEN_AI_API_KEY_30')
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

    with open(PROCESS_LOG_PATH, "w") as f:
        f.write("")

    guard_rails, reasonings = applyTopicalGuardails(query)
    if should_abort(cancel_event):
        return
    
    resp = ''
    additionalQuestions = None
    addn_questions = []
    
    if guard_rails:
        if should_abort(cancel_event):
            return
        if not IS_RAG:
            logger.info("Running without internal docs context")
            if should_abort(cancel_event):
                return
            query_type = classifierAgent(query).lower()
            if query_type == "complex":
                logger.info("Running complex task pipeline")
                if should_abort(cancel_event):
                    return
                
                # plan -> dict
                plan = plannerAgent(query)
                if should_abort(cancel_event):
                    return
                #This is the dictionary for UI Graph Construction
                dic_for_UI_graph = makeGraphJSON(plan['sub_tasks'])
                logger.debug("Graph payload: %s", dic_for_UI_graph)
                await asyncio.sleep(1)
                if should_abort(cancel_event):
                    return
                await websocket.send(json.dumps({"type": "graph", "response": json.dumps(dic_for_UI_graph)}))
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
                    tools_list = plan['sub_tasks'][sub_task]['tools']
                    agent_state = 'vanilla'
                    logger.info("Processing agent: %s", agent_name)
                    agent = Agent(sub_task, agent_name, agent_role, local_constraints, task,dependencies, tools_list, agent_state, thread_id=thread_id, model=model)
                    agentsList.append(agent)
                
                # Execute the task results using the Smack agent
                smack = Smack(agentsList)
                taskResultsDict = smack.executeSmack()
                if should_abort(cancel_event):
                    return
                for task in taskResultsDict:
                    out_str += f'{taskResultsDict[task]} \n'
                resp = drafterAgent_vanilla(query, out_str)
                if should_abort(cancel_event):
                    return
                resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
                # resp = generate_chart(resp)
                additionalQuestions = await genQuestionSimple(addn_questions)
                if should_abort(cancel_event):
                    return


            elif query_type == "simple":
                logger.info("Running simple task pipeline")
                async def executeSimplePipeline(query):
                    tools_list = [web_search_simple]
                    resp = conciseAns_vanilla(query, tools_list, thread_id=thread_id)   
                    resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
                    return str(resp)

                async def run_parallel(query):
                    resp, additionalQuestions = await asyncio.gather(
                        executeSimplePipeline(query),
                        genQuestionSimple(query)
                    )
                    return (str(resp), additionalQuestions)
                # resp = await executeSimplePipeline(query)
                resp, additionalQuestions = await run_parallel(query)
                if should_abort(cancel_event):
                    return

        elif IS_RAG:
            logger.info("Running internal docs RAG")
            rag_context = ragAgent(query, state = "concise")
            if should_abort(cancel_event):
                return
                query_type = classifierAgent_RAG(query, rag_context).lower()
                logger.info("RAG query type: %s", query_type)
            
            if query_type == "complex":
                agent_state = 'RAG'
                logger.info("Running complex task pipeline")

                rag_context = ragAgent(query, state = 'report')
                if should_abort(cancel_event):
                    return
                plan = plannerAgent_rag(query, rag_context)
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
                    tools_list = plan['sub_tasks'][sub_task]['tools']
                    logger.info("Processing agent: %s", agent_name)
                    agent = Agent(sub_task, agent_name, agent_role, local_constraints, task,dependencies, tools_list, agent_state, thread_id=thread_id, model=model)
                    agentsList.append(agent)
                
                # Execute the task results using the Smack agent
                smack = Smack(agentsList)
                taskResultsDict = smack.executeSmack()
                if should_abort(cancel_event):
                    return
                for task in taskResultsDict:
                    out_str += f'{taskResultsDict[task]} \n'
                resp = drafterAgent_rag(query,rag_context, out_str)
                if should_abort(cancel_event):
                    return
                resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
                # resp = generate_chart(resp)
                additionalQuestions = await genQuestionSimple(addn_questions)
                if should_abort(cancel_event):
                    return

                    
                with open(OUTPUT_DIR / 'drafted_response.md', 'w') as f:
                    f.write(str(resp))

            elif query_type == 'simple':
                logger.info("Running simple task pipeline")
                resp = rag_context
                resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
                additionalQuestions = await genQuestionSimple(query)
                if should_abort(cancel_event):
                    return
                
        await asyncio.sleep(1)
        if should_abort(cancel_event):
            return
        await websocket.send(json.dumps({"type": "response", "response": resp}))
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

    else:
        for key in reasonings:
            resp += f'''**{key}**\n\n'''
            resp += f'''{reasonings[key]}\n\n'''
        with open(BAD_QUESTION_PATH, "w") as f:
            f.write(resp)
        logger.info("Total time: %.2fs", time.time() - now)
        await asyncio.sleep(1)
        if should_abort(cancel_event):
            return
        await websocket.send(json.dumps({"type": "response", "response": resp}))

async def handle_connection(websocket):
    """
    Manages incoming WebSocket connections and routes messages to the appropriate handlers.
    
    Args:
        websocket (WebSocket): The WebSocket connection to manage.
    
    Returns:
        None: The function processes messages asynchronously and sends back responses to the client.
    """
    rag = False
    active_task = None
    active_cancel_event = None
    active_response_id = None
    connection_thread_id = f"conn-{uuid.uuid4().hex}"

    async def start_query(data):
        nonlocal active_task, active_cancel_event, active_response_id
        nonlocal connection_thread_id
        if active_task and not active_task.done():
            if active_cancel_event:
                active_cancel_event.set()
            active_task.cancel()

        thread_id = data.get("thread_id")
        if thread_id:
            connection_thread_id = str(thread_id)
        else:
            thread_id = connection_thread_id

        active_cancel_event = asyncio.Event()
        active_response_id = data.get("response_id")

        async def runner():
            try:
                await mainBackend(
                    data['query'],
                    websocket,
                    rag,
                    model=data.get('model', 'gpt-4o-mini'), # Default to gpt-4o-mini if not provided
                    cancel_event=active_cancel_event,
                    thread_id=thread_id,
                )
            except asyncio.CancelledError:
                return

        active_task = asyncio.create_task(runner())

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
            else:
                rag = not rag

        if data['type'] == 'cred':
                logger.info("Received credentials update")
                env_file_path = BASE_DIR / ".env"
                with open(env_file_path, 'r') as fp:
                    env_content = fp.readlines()

                env_dict = {}
                for line in env_content:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        value = value.strip('"')
                        env_dict[key] = value

                # Update the env_dict with new values from formData
                form_data = data['formData']
                for key, value in form_data.items():
                    if value:  # Only update if the value is not empty
                        if key in env_dict:
                            logger.info("Updating API key for %s", key)
                        else:
                            logger.info("Adding new API key for %s", key)
                        env_dict[key] = value

                # Write the updated environment variables back to the .env file
                with open(env_file_path, 'w') as fp:
                    for key, value in env_dict.items():
                        fp.write(f"{key}=\"{value}\"\n")

                os.environ.update(env_dict)
                try:
                    import LLMs
                    LLMs.reload_llms()
                except Exception as exc:
                    logger.exception("Failed to reload LLMs")

                logger.info(".env file has been updated.")

async def main():
    """       
     Starts the WebSocket server and keeps it running indefinitely.
    """
    logger.info("WebSocket server starting on ws://0.0.0.0:8080")
    async with websockets.serve(handle_connection, "0.0.0.0", 8080):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")
