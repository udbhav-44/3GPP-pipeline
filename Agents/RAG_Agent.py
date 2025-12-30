""""This file contains the code for the RAG Agent, which utilizes the RAG model to generate responses to the user query."""

import os
import json
import logging
from dotenv import load_dotenv
load_dotenv('.env')
logger = logging.getLogger(__name__)

from datetime import datetime
import google.generativeai as genai

from langchain.globals import set_verbose
set_verbose(os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true")

from Agents.LATS.OldfinTools import *

from LLMs import conversation_complex, GPT4o_mini_Complex



def clean(text):
    return text[text.index('{'):text.rfind('}')+1]


def ragAgent(query, state):
    fin_context = ''''''
    
    if state == "report":
        rag_result = retrieve_documents.invoke(query)
        fin_context += f'{rag_result} \n'
        sys_prompt =  '''
        Extract the Key Words, Jargons and Important Concepts from the information given below and make queries for further research:
        {
            "query_1": "...",
            "query_2": "...",
            "query_3": "..."
        }
        Following is the information to be used:
        \n
        '''
        rag_result_str = ''
        if type(rag_result) is list:
            for i in rag_result:
                if isinstance(i, str):
                    rag_result_str += i
                elif isinstance(i, dict):
                    url = i.get("url", "")
                    content = i.get("content", "")
                    rag_result_str += f"{url}+{content}"
        elif type(rag_result) is str:
            rag_result_str = rag_result
        else:
            logger.debug("Unexpected rag_result type: %s", type(rag_result))
            

        prompt = f"""Note: The Current Date and Time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. All your searches and responses must be with respect to this time frame""" + sys_prompt + rag_result_str
        response = GPT4o_mini_Complex.invoke(f'''{prompt}''').content

        dic =  dict(json.loads(clean(response.split("```")[-2].split("json")[1])))
        for p in dic:
            rag_resp = retrieve_documents.invoke(dic[p])
            fin_context += f'{rag_resp} \n'



        return fin_context
        
    elif state == "concise":
        logger.info("Running concise RAG")
        #return json.dumps(simple_query_documents.invoke(query))
        resp = simple_query_documents.invoke(query)
        if type(resp) == str:
            return resp
        elif type(resp) == dict:
            return resp['answer']
        return str(resp)
