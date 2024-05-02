from utils.extract_quetions import extract_questions
from typing_extensions import TypedDict
from typing import List
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.schema import Document
from llm.langgraph.csv_user_question.llm_model import GROQ_LLM

from langchain_groq import ChatGroq
import logging
from utils.logger import logger


GROQ_LLM = ChatGroq(
            model="llama3-70b-8192", temperature=0
        )

def route_to_research(state):
    """
    Route question and answer to web search or not.
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """

    logger.info("---ROUTER TO RESEARCH OR REWRITE---")
    df = state["df"]
    input_data = state["input_data"]
    question = state["question"]
    answer = state["answer"]

    research_router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert at understanding if the given answer need a web search to be integrated with more information. \n

        Use the following criteria to decide how to route the answer: \n\n

        If the given DATAFRAME, INPUT_DATA, QUESTION are enough to give this ANSWER.
        Just choose 'web search' if you think that the answer need more information that are not currently present, otherwise
        choose 'rewrite_answer'

        Give a binary choice 'web search' or 'rewrite_answer' based on the data provided. Return a JSON with a single key 'router_decision' and
        no premable or explaination. Use DATAFRAME, INPUT_DATA, QUESTION, ANSWER to make your decision.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        DATAFRAME: {df} \n
        INPUT_DATA: {input_data} \n
        QUESTION: {question} \n
        ANSWER: {answer}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["df","input_data", "question", "answer"],
    )

    research_router = research_router_prompt | GROQ_LLM | JsonOutputParser()


    router = research_router.invoke({"df": df,"input_data": input_data, "question": question, "answer": answer })
    if router['router_decision'] == 'web_search':
        logger.info("---ROUTE TO WEB SEARCH---")
        return "research_info"
    elif router['router_decision'] == 'rewrite_answer':
        logger.info("---ROUTE TO REWRITE ANSWER---")
        return "rewrite_answer"