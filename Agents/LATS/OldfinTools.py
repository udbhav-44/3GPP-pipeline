"""
 This file contains list of all tools that can be used by the agents. 
"""
from typing import Type , Dict , List , Union, Tuple, Callable, Any, Optional, Annotated
from pydantic import BaseModel, Field
import wikipedia
import requests
from langchain_google_community import GoogleSearchAPIWrapper
import json
import re
import os 
from datetime import datetime
from dotenv import load_dotenv
# import dotenv
import logging
from google.auth.transport.requests import Request
from langchain.tools import BaseTool, tool
import time
import urllib.parse 
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from index_3gpp_spec import *
import subprocess
import ftplib as FTP
import zipfile
from collections import defaultdict


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

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# Initialize Tavily search tool once
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
    api_url = f'http://4.188.110.145:3000/{url}'
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
    
    
@tool
def get_3gpp_docs(query: str) -> Union[Dict, str]:
    """
    Use this tool to dynamically fetch 3GPP Technical Specifications (TS) and Technical Reports (TR) from 3GPP FTP server based on the query provided.
    The docs will be saved in the temp_rag_space for indexing and further use.
    Each specification will be saved as a separate file.
    Args:
        query (str): The query related to 3GPP specifications
    Returns:
        Union[Dict, str]: Metadata of fetched TS or TR or an error message if the fetch fails.
    """
    BASE_ADDRESS = 'www.3gpp.org'
    BASE_PATH = '/Specs/archive'
    OUTPUT_DIR = 'temp_rag_space'

    index_path = "faiss_index/index_hnsw.faiss" 
    meta_path = "faiss_index/index_hnsw.meta.json"
    
    top_5 = get_top_5_specs(query,index_path, meta_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extract the "spec_no" entry from each dictionary in the list
    spec_no_list = [spec["spec_no"] for spec in top_5_specs]
    # spec_list = defaultdict(spec_no_list)
    
    try:
        for spec_no in spec_no_list:
            series, doc_number = spec_no.split('.')

            ftp = FTP(BASE_ADDRESS)
            ftp.login()
            ftp.cwd(BASE_PATH + f"/{series}_series/{series}.{doc_number}")

            filenames = ftp.nlst()
            filenames.sort()

            # Find the latest major version
            latest_major_version = None
            for filename in filenames:
                if "-" in filename:
                    major_version = filename.split('-')[1][0]
                    if not latest_major_version or major_version > latest_major_version.split('-')[1][0]:
                        latest_major_version = filename

            if not latest_major_version:
                print(f"No valid files found for spec {spec_no}. Skipping...")
                continue

            # Download the latest major version
            zip_filename = latest_major_version
            output_path = os.path.join(OUTPUT_DIR, zip_filename)

            if glob.glob(os.path.join(OUTPUT_DIR, f"{zip_filename.split('.zip')[0]}*")):
                print(f"Artifact of {zip_filename} exists. Skipping...")
                continue

            with open(output_path, "wb") as fp:
                ftp.retrbinary(f"RETR {zip_filename}", fp.write)
                print(f"{zip_filename} downloaded")

            # Unzip the file
            unzip_dir = os.path.join(OUTPUT_DIR, zip_filename.replace(".zip", ""))
            os.makedirs(unzip_dir, exist_ok=True)
            os.system(f"unzip -o {output_path} -d {unzip_dir}")

            # Delete the zip file after extraction
            os.remove(output_path)
            print(f"{zip_filename} unzipped and removed.")

        return {"status": "success", "message": "Specifications fetched and processed successfully."}

    except Exception as e:
        log_erorr(
            tool_name="3gpp_fetch",
            error_message =str(e),
            additional_info = {"query": query}
        )
        return ''

    
    
@tool    
def arxiv_fetch(query: str) -> Union[Dict, str]:
    """
    Use this tool to dynamically fetch research papers from arXiv based on the query provided.
    The papers will be saved in the temp_rag_space for indexing and further use.
    Each paper will be saved as a separate file.
    Args:
        query (str): The research topic or keywords to search for.
    Returns:
        Union[Dict, str]: Metadata of fetched papers or an error message if the fetch fails.
    """
    arxiv_api_url = f'https://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results=5'
    headers = {
        'Accept': 'application/json',
    }
    output_folder = 'temp_rag_space'

    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        response = requests.get(arxiv_api_url, headers=headers)
        response.raise_for_status()

        # Parse the response for individual papers
        papers = []
        for index, entry in enumerate(re.findall(r'<entry>(.*?)</entry>', response.text, re.DOTALL)):
            homepage_match = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
            homepage = homepage_match.group(1).strip() if homepage_match else None
            title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL).group(1).strip()
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL).group(1).strip()
            pdf_link = homepage.replace("abs","pdf") if homepage else None
            authors = re.findall(r'<author>(.*?)</author>', entry, re.DOTALL)

            paper_data = {
                "homepage" : homepage,
                "title": title,
                "summary": summary,
                "pdf_link": pdf_link,
                "authors": authors,
            }
            papers.append(paper_data)

            # Save each paper to a separate file
            filename = f"arxiv_paper_{index + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(output_folder, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=4)

        return {"status": "success", "message": "Research papers saved", "papers": papers}

    except requests.RequestException as e:
        log_error(
            tool_name="arxiv_fetch",
            error_message=str(e),
            additional_info={"query": query}
        )
        return "Error fetching papers from arXiv."

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
            "http://4.188.110.145:4005/generate",
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
            "http://4.188.110.145:4005/generate",
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
            "http://4.188.110.145:4006/v1/retrieve",
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

