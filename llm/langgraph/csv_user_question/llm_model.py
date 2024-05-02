from langchain_groq import ChatGroq


GROQ_LLM = ChatGroq(
            model="llama3-70b-8192", temperature=0
        )