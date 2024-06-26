from pydantic import BaseModel, ValidationError, Field
from typing import Optional
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import json
import ollama
from dotenv import load_dotenv, find_dotenv
import os
import streamlit as st

load_dotenv(find_dotenv())
class LLMConfig(BaseModel):
    llm_provider: str
    model: Optional[str] = Field(default=None)
    temperature: float = 0.0  

def get_llm(llm_config: Optional[LLMConfig]):

    if llm_config.llm_provider == 'Groq':
        return ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model="llama3-70b-8192", temperature=llm_config.temperature)
    elif llm_config.llm_provider == 'Ollama':
        return ChatOllama(base_url=os.getenv("OLLAMA_HOST"), model=llm_config.model, temperature=llm_config.temperature)
    elif llm_config.llm_provider == 'OpenAI':
        return ChatOpenAI(model="gpt-4o", temperature=llm_config.temperature)

def get_ollama_models():
    model_dict = ollama.list()
    return [model["name"] for model in model_dict["models"]]

def get_openai_models():
    return ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]