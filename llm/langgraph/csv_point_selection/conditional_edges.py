from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from utils.logger import logger

def summary_reflection_router(state, llm):
    """
    Route summary to critical reflection or END.
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """

    logger.info("---ROUTE TO SUMMARY OR END---")
    critics_steps = int(state["critics_steps"])
    
    if critics_steps == 2: # steps to take for self reflection
        return END
    else:
        return "summarize_critics"