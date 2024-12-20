"""
 This file contains list of all tools that can be used by the agents. 
"""
from typing import Type , Dict , List , Union, Tuple, Callable, Any, Optional, Annotated
from pydantic import BaseModel, Field
import wikipedia
import requests
import google.generativeai as genai
from langchain_google_community import GoogleSearchAPIWrapper
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from pandas import DataFrame
import dotenv
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
import os.path
from langchain.tools import BaseTool, tool
import time
import urllib.parse 
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from typing import Dict

ERROR_LOG_FILE = "./error_logs.log"
load_dotenv('.env')

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

os.environ["GOOGLE_API_KEY"]= os.getenv('GEMINI_API_KEY_30')
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_API_KEY_30')
os.environ["GOOGLE_CSE_ID"] =  os.getenv("GOOGLE_CSE_ID_30")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY_30")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY_30")
# os.environ["reddit_client_id"]= os.getenv('reddit_client_id_30')
# os.environ["reddit_client_secret"] = os.getenv('reddit_client_secret_30')
# os.environ["reddit_user_agent"] = os.getenv('reddit_user_agent_30')
# finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY_30'))
# os.environ["DISCORD_AUTH_KEY"] = os.getenv('DISCORD_AUTH_KEY')

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

from langchain_community.tools import TavilySearchResults

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
    
    # JINA HOSTING - URL Change
    api_url = f'http://127.0.0.1:3000/{url}'
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

search = GoogleSearchAPIWrapper()
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
    tavily_search = TavilySearchResults(
        max_results=2,
        search_depth="basic",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )
    try:
        res = []
        search_results = tavily_search.invoke({"query": query})
        try:
            for search_result in search_results:
                url = search_result['url']
                content = search_result['content']
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
    tavily_search = TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )
    
    try:
        search_results = tavily_search.invoke({"query": query})
        return search_results
            
    except Exception as e:
        # If both fail, return error message
        log_error(
            tool_name="tavily_web_search_simple",
            error_message=str(e),
            additional_info={"query": query}
        )
        return ''
    

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
        print("Started")
        start = time.time()
        
        payload = {
            "query": prompt,  # No need to quote the prompt
            "source": source  # source should be a string, not a set
        }
        
        response = requests.post(
            "http://localhost:4005/generate",
            headers={"Content-Type": "application/json"},

            json=payload

        )
        
        print(f"Response status code: {response.status_code}")
        print("Posted")
        response.raise_for_status()  # Raise an error for HTTP issues
        print("Raised")
        
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        
        result = response.json()
        print(result)
        return result
    
    except requests.RequestException as e:
        print(f"HTTP Request failed: {e}")
        if hasattr(e, 'response'):
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        log_error(
            tool_name="query_documents",
            error_message=str(e),
            additional_info={"prompt": prompt, "source": source}
        )
        return "This tool is not working right now. DO NOT CALL THIS TOOL AGAIN!"


@tool
def get_wikipedia_summary(query: str):
    """
    Fetches a summary from Wikipedia based on a search query.
    Args:
        query (str): The search query terms to look up on Wikipedia. Also there should be less than four terms.

    Returns:
        str: A summary of the Wikipedia page found for the query. If no results are found,
             or if there is an error fetching the page, appropriate messages are returned.
    """
    
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return web_search_simple.invoke(query)
        try:
            result = wikipedia.page(search_results[0])
            return f"Found match with {search_results[0]}, Here is the result:\n{result.summary}"
        except:
            result = wikipedia.page(search_results[1])
            return f"Found match with {search_results[1]}, Here is the result:\n{result.summary}"
    except Exception as e:
        log_error(
            tool_name="get_wikipedia_summary",
            error_message=str(e),
            additional_info={"query": query}
        )
        
        return "This tool is not working right now. DO NOT CALL THIS TOOL AGAIN!"
    

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
        print("Started_simple")
        start = time.time()
        
        payload = {
            "query": prompt, 
            "destination": 'user' 
        }
        print(payload)
        response = requests.post(
            "http://localhost:4005/generate",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        print(f"Response status code: {response.status_code}")
        print("Posted_simple")
        response.raise_for_status()  # Raise an error for HTTP issues
        print("Raised_simple")
        
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        
        result = response.json()
        print(result)
        return result
    
    except requests.RequestException as e:
        print(f"HTTP Request failed: {e}")
        if hasattr(e, 'response'):
            if e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
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
        print("Started")
        start = time.time()
        
        payload = {
            "query": prompt,
            "k" : 2  , 
            "destination" : 'user'
        }
        
        response = requests.post(
            "http://localhost:4006/v1/retrieve",
            headers={"Content-Type": "application/json"},

            json=payload

        )
        
        print(f"Response status code: {response.status_code}")
        print("Posted")
        response.raise_for_status()  # Raise an error for HTTP issues
        print("Raised")
        
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        
        result = response.json()
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
        print(f"HTTP Request failed: {e}")
        if hasattr(e, 'response'):
            if e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
        log_error(
            tool_name="query_documents",
            error_message=str(e),
            additional_info={"prompt": prompt}
        )
        return web_search_simple.invoke(prompt)


        
        