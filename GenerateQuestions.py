"""
This file provides functions to generate complex and simple questions based on user prompts.
"""
import os
import logging
from dotenv import load_dotenv
import json
from LLMs import GPT4o_mini_Complex

load_dotenv('.env')
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv('OPEN_AI_API_KEY_30')


async def genQuestionSimple(query):
    """ 
    Synthesizes five detailed questions related to a given query.
        Args:
            query (str): The main query for which questions need to be generated.
        Returns:
            list: A list of five synthesized questions.
    """

    logger.info("Executing genQuestionSimple")
    system_prompt = '''
       Synthesize 5 Questions Related to the given query with a focus on communication systems and 3GPP technical standards.
        Instructions to make the questions:
        - The questions should focus on topics related to 3GPP technical specifications, mobile communication systems, protocols, and network architectures.
        - Questions should encourage detailed answers, requiring in-depth research and reasoning.
        - Make the questions diverse but ensure they remain relevant to the core topics of 3GPP and communication technologies.
        - The questions should be *at least* 20 words long.
        - Output the questions according to the schema:
        {
        "question1": "...",
        "question2": "...",
        "question3": "...",
        "question4": "...",
        "question5": "..."
        }
    '''

    user_prompt = f'''
    The User Prompt is

    {query}
    '''

    prompt = f'''{system_prompt}\n\n {user_prompt}'''
    response = GPT4o_mini_Complex.invoke(f'''{prompt}''').content

    logger.info("Executed genQuestionSimple")
    return json.loads(response).values()
