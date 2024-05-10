from langchain_groq import ChatGroq
from langchain_community.llms import Ollama


GROQ_LLM = Ollama(
            model="llama3:8b", temperature=0
            # model="llama3-8b-8192", temperature=0
            )