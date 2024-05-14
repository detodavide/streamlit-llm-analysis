from utils.extract_quetions import extract_questions
from typing_extensions import TypedDict
from typing import List
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from utils.logger import logger

def generate_questions(state, llm):   
    """Take the initial df and input_data to generate the questions based on the data"""
    logger.info("---GENERATING THE QUESTIONS---")
    df = state["df"]
    input_data = state["input_data"]
    num_steps = int(state["num_steps"])
    num_steps += 1

    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a Data Analyst Agent that is an expert on making insightful questions.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Given the following dataframe: {df}\n
    Write 20 questions that focus on the INPUT DATA in relation to the whole dataframe ( correlations, meaningful insights ...), no preamble or explanation.

    INPUT DATA:{input_data}\n\n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["df", "input_data"],
    )

    questions_generator = prompt | llm | StrOutputParser()
    questions = questions_generator.invoke({"df": df, "input_data": input_data})

    questions = extract_questions(questions)
    logger.info(f"Generated questions: {questions}")

    return ({"questions": questions, "num_steps": num_steps})

def questions_answering(state, llm):
    """Given any number of questions answer them"""
    logger.info("---ANSWERING QUESTIONS---")
    df = state["df"]
    input_data = state["input_data"]
    num_steps = int(state["num_steps"])
    questions = state["questions"]
    num_steps += 1

    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a Data Analyst expert that is able to find meaningful insights answering the user questions about data.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        DATAFRAME: {df}\n
        INPUT DATA: {input_data}\n
        QUESTIONS: {questions}\n

        Answer each questions giving a proper analysis and explanation on how the INPUT DATA compares to the whole DATAFRAME, giving a strong focues on the INPUT DATA.
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["df", "input_data", "questions"],
    )

    answers_generator = prompt | llm | StrOutputParser()
    answers = answers_generator.invoke({"df": df, "input_data": input_data, "questions": questions})
    logger.info(f"Generated answers: {answers}")

    return ({"answers": answers, "num_steps": num_steps})

def summarize_answers(state,llm):
    """Summarize the given answers"""
    logger.info("---SUMMARIZING ANSWERS---")
    answers = state["answers"]
    summary_critics = state["summary_critics"]
    input_data = state["input_data"]
    num_steps = int(state["num_steps"])
    num_steps += 1
    
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a Data Analyst expert that is a master on summarize informations and return it in italian only.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        ANSWERS: {answers}\n

        Summarize the whole answers in a very discorsive single paragraph without adding any preamble or introduction.
        Just answer with the summary in a non-technical way like you are talking to the common user.

        You can use this critics if present, to correct your output:
        CRITICS: {summary_critics}\n
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["answers", "summary_critics"],
    )

    summary_generator = prompt | llm | StrOutputParser()
    summary = summary_generator.invoke({"answers": answers, "summary_critics": summary_critics})
    logger.info(f"Generated questions: {summary}")

    return ({"summary": summary, "num_steps": num_steps, "summary_critics": summary_critics})

def summarize_critics(state, llm):
    """Give critical opinions on the summarization"""
    logger.info("---SUMMARIZING ANSWERS---")
    answers = state["answers"]
    summary = state["summary"]
    num_steps = int(state["num_steps"])
    critics_steps = int(state["critics_steps"])
    input_data = state["input_data"]

    num_steps += 1
    critics_steps += 1

    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert on critic a text and giving useful critical insights.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Given a SUMMARY on some ANSWERS, return a insightful critics to better the SUMMARY, remind that the summary must be short and aknowledge the focus on the INPUT DATA.\n

        SUMMARY: {summary}\n
        ANSWERS: {answers}\n
        INPUT DATA: {input_data}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["answers", "summary", "input_data"],
    )

    critics_reflection = prompt | llm | StrOutputParser()
    summary_critics = critics_reflection.invoke({"answers": answers, "summary": summary, "input_data": input_data})
    logger.info(f"Generated questions: {summary_critics}")

    return ({"num_steps": num_steps, "summary_critics": summary_critics, "critics_steps": critics_steps})