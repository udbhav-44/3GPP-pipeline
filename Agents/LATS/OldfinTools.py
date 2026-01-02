"""
 This file contains list of all tools that can be used by the agents. 
"""
from typing import Type , Dict , List , Union, Tuple, Any, Optional
from pydantic import BaseModel, Field
import requests
import json
import re
import os 
from datetime import datetime
from dotenv import load_dotenv
import logging
from langchain.tools import BaseTool, tool
import time
import urllib.parse 
import zipfile
import traceback
import shutil
from neo4j import GraphDatabase
import pandas as pd
from spire.doc import Document, FileFormat
import concurrent.futures


ERROR_LOG_FILE = "./error_logs.log"

# Step 1: Create a logger
logger = logging.getLogger('my_logger')
file_Handler = logging.FileHandler(ERROR_LOG_FILE)
logger.setLevel(logging.DEBUG)  # Set the base logging level
file_Handler.setLevel(logging.ERROR)  # Set the handler logging level
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.addHandler(file_Handler)
def log_error(tool_name, error_message, additional_info=None):
    error_entry = {
        "tool" : tool_name,
        "error_message" : error_message,
        "timestamp" : datetime.now().isoformat(),
        "additional info" : additional_info or {}
    }
    logger.error(json.dumps(error_entry, indent=4))

PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PIPELINE_ROOT, ".env"), override=False)
RESULTS_CSV_PATH = os.path.join(PIPELINE_ROOT, "Results.csv")
RESULTS_COLUMNS = [
    "doc_id",
    "title",
    "source_path",
    "meeting_id",
    "release",
    "total_score",
    "boosted_score",
]
JINA_SCRAPE_URL = os.getenv("JINA_SCRAPE_URL", "http://wisdomlab3gpp.live:3000")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://wisdomlab3gpp.live:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "login123")
RAG_GENERATE_URL = os.getenv("RAG_GENERATE_URL", "http://wisdomlab3gpp.live:4005/generate")
RAG_STATS_URL = os.getenv("RAG_STATS_URL", "http://wisdomlab3gpp.live:4004/v1/statistics")
RAG_RETRIEVE_URL = os.getenv("RAG_RETRIEVE_URL", "http://wisdomlab3gpp.live:4006/v1/retrieve")
RAG_UPLOADS_DIR = os.getenv("RAG_UPLOADS_DIR", "/git_folder/udbhav/code/RAG/uploads")

import contextvars
# Context variable to store the current model. 
# This must be set by the caller (Agent/SolveSubQuery) before invoking tools.
current_model_var = contextvars.ContextVar("current_model", default="gpt-4o-mini")
current_user_var = contextvars.ContextVar("current_user_id", default=None)

def get_current_model():
    return current_model_var.get()

def set_current_model(model):
    return current_model_var.set(model)

def reset_current_model(token):
    current_model_var.reset(token)


def get_current_user_id():
    return current_user_var.get()


def set_current_user_id(user_id):
    return current_user_var.set(str(user_id) if user_id else None)


def reset_current_user_id(token):
    if token is not None:
        current_user_var.reset(token)



def save_results_csv(df, csv_path=RESULTS_CSV_PATH):
    tmp_csv_path = f"{csv_path}.tmp"
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(tmp_csv_path, index=False)
        os.replace(tmp_csv_path, csv_path)
        logging.info(f"Saved results atomically to {csv_path}")
        return True
    except Exception as e:
        log_error("search_and_generate:csv_save", str(e), {"csv_path": csv_path})
        try:
            if os.path.exists(tmp_csv_path):
                os.remove(tmp_csv_path)
        except OSError:
            pass
        return False


os.environ["GOOGLE_API_KEY"]= os.getenv('GEMINI_API_KEY_30')
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_API_KEY_30')
os.environ["GOOGLE_CSE_ID"] =  os.getenv("GOOGLE_CSE_ID_30")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY_30")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY_30")

TAVILY_API_URL = "https://api.tavily.com/search"

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def _tavily_search_request(
    query: str,
    max_results: int,
    search_depth: str,
    include_answer: bool,
    include_raw_content: bool,
    include_images: bool,
) -> List[Dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        log_error(
            tool_name="tavily_web_search",
            error_message="Missing TAVILY_API_KEY",
            additional_info={"query": query},
        )
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "include_images": include_images,
    }

    try:
        response = requests.post(TAVILY_API_URL, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except requests.RequestException as e:
        log_error(
            tool_name="tavily_web_search",
            error_message=f"{type(e).__name__}: {e}",
            additional_info={
                "query": query,
                # "status_code": getattr(getattr(e, "response", None), "status_code", None),
            },
        )
        return []
    except Exception as e:
        log_error(
            tool_name="tavily_web_search",
            error_message=f"{type(e).__name__}: {e}",
            additional_info={"query": query},
        )
        return []

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, you agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
    

@tool
def web_scrape(url, query) -> Union[Dict, str]:
    """
    Use this to scrape a web page using links found using web search to give detailed response. Input should be the URL of the page to scrape.
    Returns the scraped data as JSON if successful, else move on to the next best site in case of errors like required login, captcha etc.
    Args:
        url (str): The URL of the page to scrape.
        query (str): The query for which the page is being scraped.
    Returns:
        Union[Dict, str]: The scraped data as JSON if successful
    """
    print("web_scrape invoked")
    # JINA HOSTING - URL Change
    api_url = f"{JINA_SCRAPE_URL.rstrip('/')}/{url}"
    headers = {
        'Accept': 'application/json',
        'X-Respond-With':'markdown',
        
    }
    output_folder = 'temp_rag_space'
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate filename based on URL and timestamp
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(output_folder, filename)

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        try:
            data = response.json()
            data_str = str(data)
        except ValueError:
            data_str = response.text
        finally:
            # Save the data to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(data_str)
            delay = 2
            time.sleep(delay)

            return query_documents.invoke({"prompt":query,"source":url})

    except requests.RequestException as e:
        log_error(
            tool_name="web_scrape",
            error_message=str(e),
            additional_info={"url": url}
        )
        return url
    
    

def list_directories(ftp_client, base_path):
    """
    Lists directories in the given base_path on the FTP server.
    """
    try:
        ftp_client.cwd(base_path)
        directories = ftp_client.nlst()
        logging.info("Available directories in %s: %s", base_path, directories)
        return directories
    except Exception as e:
        logging.exception("Error listing directories in %s", base_path)
        traceback.print_exc()
        return []
    
def clear_directory(path):

    """Clear all files and directories in the specified path"""
    logging.info(f"Clearing directory: {path}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def format_response(response_text):
    """Format JSON or text responses nicely"""
    logging.info(f"Formatting response: {response_text}")
    try:
        data = json.loads(response_text)
        formatted = "### Generated Response\n\n"
        if isinstance(data, dict):
            for k, v in data.items():
                formatted += f"**{k}**:\n{v}\n\n"
        elif isinstance(data, list):
            for i, item in enumerate(data, start=1):
                formatted += f"**Result {i}:**\n{json.dumps(item, indent=2)}\n\n"
        else:
            formatted = str(data)
        return formatted
    except json.JSONDecodeError:
        # Fallback: raw text as markdown
        return f"###  Generated Response\n\n{response_text}"


@tool
def search_and_generate(query_str: str, meeting_id: Optional[str] = "") -> Tuple[Optional[Any], str]:
    """
    Tool to get relevant documents from Graph3GPP based on a search query and generate a response using RAG service.
    Returns a tuple of (DataFrame of results or None, formatted response string).
    meeting_id is optional and can be used to filter documents by meeting.

    USE THIS TOOL INSTEAD OF query_documents FOR BEST RESULTS. IN CASE OF FAILURE, FALLBACK TO query_documents.

    Args:
        query_str (str): The search query string.
        meeting_id (Optional[str]): An optional meeting ID to filter documents.
    Returns:
        Tuple[Optional[Any], str]: A tuple containing a DataFrame of results (or None) and a formatted response string.

    CALL THIS TOOL FIRST BEFORE ANY OTHER TOOL.
    """
    output_dir = "downloaded_docs"
    uri = NEO4J_URI
    uploads_dir = RAG_UPLOADS_DIR
    generate_uri = RAG_GENERATE_URL
    stats_uri = RAG_STATS_URL
    uname = NEO4J_USER
    pswd = NEO4J_PASSWORD

    logging.info(f"Received search request: {query_str}, {meeting_id}")

    try:

        os.makedirs(output_dir, exist_ok=True)
        clear_directory(output_dir)
        clear_directory(uploads_dir)

        query = """
        CALL () {
          CALL db.index.fulltext.queryNodes("docIndex", $query)
          YIELD node, score
          WHERE $meeting IS NULL OR node.meeting_id CONTAINS $meeting
          RETURN
            collect(node.doc_id) AS direct_doc_ids,
            collect({doc_id: node.doc_id, score: score}) AS direct_docs
        }
        WITH direct_doc_ids, direct_docs
        CALL (direct_doc_ids) {
          WITH direct_doc_ids
          CALL db.index.fulltext.queryNodes("agendaIndex", $query)
          YIELD node, score AS agenda_score
          MATCH (node)<-[:APPEARS_IN]-(d:Document)
          WITH d,
               CASE
                 WHEN d.doc_id IN direct_doc_ids THEN agenda_score * 2.3
                 ELSE agenda_score * 0.8
               END AS agenda_rel_score
          RETURN collect({doc_id: d.doc_id, score: agenda_rel_score}) AS agenda_docs
        }
        WITH direct_docs, agenda_docs
        CALL() {
          CALL db.index.fulltext.queryNodes("techEntityIndex", $query)
          YIELD node, score AS entity_score
          MATCH (d:Document)-[:MENTIONS]->(node)
          RETURN collect({doc_id: d.doc_id, score: entity_score * 0.7}) AS entity_docs
        }
        WITH direct_docs, agenda_docs, entity_docs
        WITH direct_docs + agenda_docs + entity_docs AS all_docs
        UNWIND all_docs AS doc_entry
        WITH doc_entry.doc_id AS doc_id, sum(doc_entry.score) AS total_score
        MATCH (d:Document {doc_id: doc_id})
        WITH d, total_score,
        CASE
          WHEN d.title CONTAINS 'Feature Lead Summary' THEN total_score * 2.0
          WHEN d.title CONTAINS 'Feature Lead' THEN total_score * 1.5
          ELSE total_score
        END AS boosted_score
        RETURN
          d.doc_id,
          d.title,
          d.source_path,
          d.meeting_id,
          d.release,
          total_score,
          boosted_score
        ORDER BY boosted_score DESC
        LIMIT 15;
        """

        driver = GraphDatabase.driver(uri, auth=(uname, pswd))
        logging.info(f"Connected to Neo4j: {uri}")

        meeting = meeting_id.strip() if meeting_id and meeting_id.strip() else None
        params = {"query": query_str, "meeting": meeting}

        with driver.session() as session:
            result = session.run(query, params)
            data = [record.data() for record in result]

        driver.close()
        logging.info(f"Found {len(data)} documents")

        if not data:
            empty_df = pd.DataFrame(columns=RESULTS_COLUMNS)
            save_results_csv(empty_df)
            return empty_df, "⚠️ No matching documents found."

        df = pd.DataFrame(data)

        def download_and_extract(row):
            url, doc_id, title = row._3, row._1, row._2[:50].replace("/", "_")
            dest_path = os.path.join(output_dir, f"{doc_id} - {title}.zip")
            temp_extract_dir = os.path.join("/tmp/extracted_docs", str(doc_id))
            os.makedirs(temp_extract_dir, exist_ok=True)

            try:
                encoded_url = urllib.parse.quote(url, safe=':/')
                r = requests.get(encoded_url, timeout=20)
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    f.write(r.content)

                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if not (member.startswith('__MACOSX/') or member.endswith('.DS_Store')):
                            zip_ref.extract(member, temp_extract_dir)
                os.remove(dest_path)

                for root, _, files in os.walk(temp_extract_dir):
                    for fname in files:
                        src_path = os.path.join(root, fname)
                        dst_path = os.path.join(uploads_dir, fname)

                        if fname.lower().endswith((".doc", ".docm")):
                            try:
                                # If Document/FileFormat aren't available this will raise and be caught
                                document = Document()
                                document.LoadFromFile(src_path)
                                if document.IsContainMacro:
                                    logging.info(f"[{fname}] contains macros — removing...")
                                    document.ClearMacros()
                                clean_path = os.path.splitext(dst_path)[0] + ".docx"
                                document.SaveToFile(clean_path, FileFormat.Docx2016)
                                document.Close()
                                logging.info(f"Cleaned and moved safely: {clean_path}")
                            except Exception as e:
                                logging.error(f"Spire.Doc conversion failed for {fname}: {e}")
                        else:
                            safe_dst = os.path.join(uploads_dir, fname)
                            shutil.move(src_path, safe_dst)

                shutil.rmtree(temp_extract_dir, ignore_errors=True)
                return (title, None)
            except Exception as e:
                logging.error(f"Error downloading {title}: {e}")
                return (title, str(e))

        download_errors = []
        max_workers = min(20, len(df))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_and_extract, row): row for row in df.itertuples()}
            for future in concurrent.futures.as_completed(futures):
                title, err = future.result()
                if err:
                    download_errors.append(f"{title}: {err}")

        logging.info(f"Downloaded {len(df) - len(download_errors)}/{len(df)} documents successfully")

        # wait for AI service readiness
        start_time = time.time()
        max_wait = 300
        ready = False
        while time.time() - start_time < max_wait:
            try:
                resp = requests.get(stats_uri, timeout=5)
                if resp.status_code == 200:
                    ready = True
                    logging.info("AI service ready.")
                    break
                else:
                    logging.info(f"AI not ready, status={resp.status_code}")
            except Exception as e:
                logging.info(f"AI service check failed: {e}")
            time.sleep(5)

        if not ready:
            msg = f"Timeout: AI service not ready after {max_wait/60:.1f} minutes."
            logging.error(msg)
            return df, msg

        # generate
        payload = {"query": query_str, "max_tokens": 5000, "num_docs": 10, "model": get_current_model()}
        try:
            response = requests.post(generate_uri, json=payload, timeout=90)
            response.raise_for_status()
            formatted_response = format_response(json.dumps(response.json()))
        except Exception as e:
            formatted_response = f"Failed to generate response: {e}"
            log_error("search_and_generate", str(e), {"query": query_str, "meeting_id": meeting_id})

        save_results_csv(df)


        return df, formatted_response

    except Exception as e:
        log_error("search_and_generate", str(e), {"query": query_str, "meeting_id": meeting_id})
        return None, f"Error: {e}"  
    



@tool
def web_search(query: str):
    """
    If you do not know about an entity, Perform web search using google search engine. 
    This should be followed by web scraping the most relevant page to get detailed response.
    Args:
        query (str): The query to search for.
    Returns:
        str: The URL of the most relevant page to scrape.
    """
    print("web_search invoked")
    try:
        res = []
        search_results = _tavily_search_request(
            query=query,
            max_results=2,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
        if not search_results:
            return ''
        try:
            for search_result in search_results:
                url = search_result.get("url")
                if not url:
                    continue
                res.append(web_scrape.invoke({"url": url, "query": query}))
            return res
        except Exception as e:
            # If both fail, return error message
            log_error(
                tool_name="tavily_web_search",
                error_message=str(e),
                additional_info={"query": query}
            )
            return search_results
    except Exception as e:
        # If both fail, return error message
        log_error(
            tool_name="tavily_web_search",
            error_message=str(e),
            additional_info={"query": query}
        )
        return ''

@tool
def web_search_simple(query: str):
    """
    If you do not know about an entity, Perform web search using google search engine.
    Args:
        query (str): The query to search for.
    Returns:
        str: The URL of the most relevant page to scrape.      
    """
    
    print("web_search_simple invoked")
    
    try:
        return _tavily_search_request(
            query=query,
            max_results=3,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
    except Exception as e:
        log_error(
            tool_name="tavily_web_search_simple",
            error_message=str(e),
            additional_info={"query": query},
        )
        return ''


def _user_path_matches(metadata: Dict[str, Any], user_id: str) -> bool:
    path = str(metadata.get("path", "")).replace("\\", "/").lower()
    token = f"/user_uploads/{user_id}/".lower()
    return token in path or path.startswith(f"user_uploads/{user_id}/".lower())


def _filter_docs_for_user(docs: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
    if not user_id:
        return docs
    filtered = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        if _user_path_matches(meta, user_id):
            filtered.append(doc)
    return filtered



@tool
def query_documents(prompt: str, source: str) -> Dict:
    """
    Query documents using a Retrieval-Augmented Generation (RAG) endpoint.
    This should be the first choice before doing web search,
    if this fails or returns unsatisfactory results, then use web search for the same query.
    Args:
        prompt (str): The prompt to send to the RAG endpoint.
        source (str): The source URL of the document.

    Returns:
        Dict: The JSON response from the RAG endpoint, containing the retrieved information and generated answer.
    """
    try:
        logging.info("query_documents started")
        start = time.time()
        
        user_id = get_current_user_id()
        payload = {
            "query": prompt,  # No need to quote the prompt
            "source": source,  # source should be a string, not a set
            "model": get_current_model()
        }
        if user_id:
            payload["user_id"] = user_id
        
        response = requests.post(
            RAG_GENERATE_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        logging.info("query_documents response status: %s", response.status_code)
        logging.info("query_documents posted")
        response.raise_for_status()  # Raise an error for HTTP issues
        logging.info("query_documents response validated")
        
        end = time.time()
        logging.info("query_documents time taken: %.2fs", end - start)
        
        result = response.json()
        logging.debug("query_documents result: %s", result)
        return result
    
    except requests.RequestException as e:
        logging.exception("query_documents request failed")
        if hasattr(e, 'response'):
            logging.error("query_documents response status: %s", e.response.status_code)
            logging.error("query_documents response content: %s", e.response.text)
        log_error(
            tool_name="query_documents",
            error_message=str(e),
            additional_info={"prompt": prompt, "source": source}
        )
        return "This tool is not working right now. DO NOT CALL THIS TOOL AGAIN!"


# @tool
# def get_wikipedia_summary(query: str):
#     """
#     Fetches a summary from Wikipedia based on a search query.
#     Args:
#         query (str): The search query terms to look up on Wikipedia. Also there should be less than four terms.

#     Returns:
#         str: A summary of the Wikipedia page found for the query. If no results are found,
#              or if there is an error fetching the page, appropriate messages are returned.
#     """
    
#     try:
#         search_results = wikipedia.search(query)
#         if not search_results:
#             return web_search_simple.invoke(query)
#         try:
#             result = wikipedia.page(search_results[0])
#             return f"Found match with {search_results[0]}, Here is the result:\n{result.summary}"
#         except:
#             result = wikipedia.page(search_results[1])
#             return f"Found match with {search_results[1]}, Here is the result:\n{result.summary}"
#     except Exception as e:
#         log_error(
#             tool_name="get_wikipedia_summary",
#             error_message=str(e),
#             additional_info={"query": query}
#         )
        
#         return "This tool is not working right now. DO NOT CALL THIS TOOL AGAIN!"



@tool
def simple_query_documents(prompt: str) -> Dict:
    """
    Query documents using a Retrieval-Augmented Generation (RAG) endpoint.
    This should be the first choice before doing web search,
    if this fails or returns unsatisfactory results, then use web search for the same query.

    Args:
        prompt (str): The prompt to send to the RAG endpoint.
        source (str): The source URL of the document.

    Returns:
        Dict: The JSON response from the RAG endpoint, containing the retrieved information and generated answer.
    """
    try:
        logging.info("simple_query_documents started")
        start = time.time()
        
        user_id = get_current_user_id()
        payload = {
            "query": prompt, 
            "destination": 'user',
            "model": get_current_model()
        }
        if user_id:
            payload["user_id"] = user_id
        logging.debug("simple_query_documents payload: %s", payload)
        response = requests.post(
            RAG_GENERATE_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        logging.info("simple_query_documents response status: %s", response.status_code)
        logging.info("simple_query_documents posted")
        response.raise_for_status()  # Raise an error for HTTP issues
        logging.info("simple_query_documents response validated")
        
        end = time.time()
        logging.info("simple_query_documents time taken: %.2fs", end - start)
        
        result = response.json()
        logging.debug("simple_query_documents result: %s", result)
        return result
    
    except requests.RequestException as e:
        logging.exception("simple_query_documents request failed")
        if hasattr(e, 'response'):
            if e.response is not None:
                logging.error("simple_query_documents response status: %s", e.response.status_code)
                logging.error("simple_query_documents response content: %s", e.response.text)
        log_error(
            tool_name="simple_query_documents",
            error_message=str(e),
            additional_info={"prompt": prompt}
        )
        
        return ''
        
@tool
def retrieve_documents(prompt: str) -> str:
    """
    Extract Information from the provided internal document
    Since this is the main source of information which is always 
    correct, this should be the first choice of tool for any agent.
    CALL THIS BEFORE CALLING ANY OTHER TOOL.
    Args:
        prompt (str): The prompt to send to the RAG endpoint.
        source (str): The source URL of the document.
    Returns:
        Dict: The JSON response from the RAG endpoint, containing the retrieved information and generated answer.
    """

    try:
        logging.info("retrieve_documents started")
        start = time.time()
        
        user_id = get_current_user_id()
        k = 2
        if user_id:
            k = 12
        payload = {
            "query": prompt,
            "k": k,
            "destination": 'user'
        }
        
        response = requests.post(
            RAG_RETRIEVE_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        logging.info("retrieve_documents response status: %s", response.status_code)
        logging.info("retrieve_documents posted")
        response.raise_for_status()  # Raise an error for HTTP issues
        logging.info("retrieve_documents response validated")
        
        end = time.time()
        logging.info("retrieve_documents time taken: %.2fs", end - start)
        
        result = response.json()
        if user_id:
            result = _filter_docs_for_user(result, user_id)
            if not result:
                return "No matching documents found for this user. Ask the user to upload relevant documents."
        out = ''
        for i in result:
            for j in i.values():
                if type(j) is str:
                    out += f"{j} "
                elif type(j) is dict:
                    for k in j.values():
                        out += f"{str(j)} "
                    out+= '\n'
            out+= '\n'
        return out
    
    except requests.RequestException as e:
        logging.exception("retrieve_documents request failed")
        if hasattr(e, 'response'):
            if e.response is not None:
                logging.error("retrieve_documents response status: %s", e.response.status_code)
                logging.error("retrieve_documents response content: %s", e.response.text)
        log_error(
            tool_name="query_documents",
            error_message=str(e),
            additional_info={"prompt": prompt}
        )
        return "This tool is not working right now. DO NOT CALL THIS TOOL AGAIN!"
        # return web_search_simple.invoke(prompt)
