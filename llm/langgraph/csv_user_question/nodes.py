from utils.extract_quetions import extract_questions
from typing_extensions import TypedDict
from typing import List
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from llm.llm_model import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.schema import Document

from langchain_groq import ChatGroq
import logging
from utils.logger import logger


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

def get_answer(state, llm):
    """Answer the user question"""
    logger.info("---ANSWERING QUESTIONS---")
    df = state["df"]
    input_data = state["input_data"]
    num_steps = int(state["num_steps"])
    question = state["question"]
    num_steps += 1

    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a Data Analyst expert that is able to find meaningful insights answering the user questions about data.
        The answer must be in italian.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        DATAFRAME: {df}\n
        INPUT DATA: {input_data}\n
        QUESTION: {question}\n

        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["df", "input_data", "questions"],
    )

    answer_generator = prompt | llm | StrOutputParser()
    answer = answer_generator.invoke({"df": df, "input_data": input_data, "question": question})
    logger.info(f"Generated answer: {answer}")

    return ({"answer": answer, "num_steps": num_steps})

def web_search(state, llm):
    """Search for more info based on the found keyword"""
    logger.info("---WEB SEARCH---")
    df = state["df"]
    input_data = state["input_data"]
    num_steps = int(state["num_steps"])
    question = state["question"]   
    answer = state["answer"] 
    num_steps += 1


    search_keyword_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a master at working out the best keywords to search for in a web search to get the best info for the customer.

        given the DATAFRAME, INPUT_DATA, QUESTION, ANSWER Work out the best keywords that will find the best
        info for helping to write the final answer.

        Return a JSON with a single key 'keywords' with no more than 3 keywords and no preamble or explanation.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        DATAFRAME: {df} \n
        INPUT_DATA: {input_data} \n
        QUESTION: {question} \n
        ANSWER: {answer}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["df","input_data", "question", "answer"],
    )

    search_keyword_chain = search_keyword_prompt | llm | JsonOutputParser()
    keywords = search_keyword_chain.invoke({"df": df, "input_data": input_data, "question": question, "answer": answer})

    keywords = keywords['keywords']
    search = DuckDuckGoSearchResults()
    full_searches = []
    for keyword in keywords[:1]:
        temp_docs = search.run(keyword)
        web_results = "\n".join([d["content"] for d in temp_docs])
        web_results = Document(page_content=web_results)
        if full_searches is not None:
            full_searches.append(web_results)
        else:
            full_searches = [web_results]
    return {"research_info": full_searches, "num_steps":num_steps}

def rewrite_answer(state, llm):
    """Rewrite the answer using the given data"""
    logger.info("---REWRITING THE FINAL ANSWER---")
    df = state["df"]
    input_data = state["input_data"]
    num_steps = int(state["num_steps"])
    question = state["question"]
    answer = state["answer"]
    research_info = state["research_info"]
    num_steps += 1

    rewrite_answer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are the Final Answer Agent read the ANSWER from answering agent and provide a simple summary in italian.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        DATAFRAME: {df} \n\n
        RESEARCH_INFO: {research_info} \n\n
        INPUT_DATA: {input_data}\n\n
        QUESTION: {question} \n\n
        ANSWER: {answer} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["df",
                        "question",
                        "input_data",
                        "research_info",
                        "answer",
                        ],
    )

    rewrite_chain = rewrite_answer_prompt | llm | StrOutputParser()

    final_answer= rewrite_chain.invoke(
                                {
                                    "df": df,
                                    "input_data":input_data,
                                    "research_info":research_info,
                                    "question": question,
                                    "answer":answer
                                }
                                )
    
    return ({"final_answer": final_answer})