from pydantic import BaseModel, ValidationError
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import json

class LLMConfig(BaseModel):
    llm_provider: str
    model: str
    temperature: float = 0.0  

def get_llm():
    try:
        with open("./llm/llm_config.json", 'r') as file:
            data = json.load(file)
            config = LLMConfig(**data)
    except (FileNotFoundError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return None

    if config.llm_provider == 'Groq':
        return ChatGroq(model=config.model, temperature=config.temperature)
    elif config.llm_provider == 'Ollama':
        return ChatOllama(model=config.model, temperature=config.temperature)
    elif config.llm_provider == 'OpenAI':
        return ChatOpenAI(model=config.model, temperature=config.temperature)

LLM = get_llm()
