from langgraph.graph import StateGraph, END
from llm.langgraph.csv_point_selection.states import GraphState as CPSGraphState, OneShotGraph
from llm.langgraph.csv_point_selection.nodes import *
from llm.langgraph.csv_point_selection.conditional_edges import *
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# Init Graph
def full_node_schema(llm: ChatGroq | ChatOllama | ChatOpenAI):
    workflow = StateGraph(CPSGraphState)

    # Nodes
    workflow.add_node("generate_questions", lambda state: generate_questions(state, llm))
    workflow.add_node("questions_answering", lambda state: questions_answering(state, llm))
    workflow.add_node("summarize_answers", lambda state: summarize_answers(state, llm))
    workflow.add_node("summarize_critics", lambda state: summarize_critics(state, llm))

    # Edges
    workflow.add_edge("generate_questions", "questions_answering")
    workflow.add_edge("questions_answering", "summarize_answers")
    workflow.add_edge("summarize_critics", "summarize_answers")
    workflow.add_conditional_edges(
            "summarize_answers",
            lambda state: summary_reflection_router(state, llm)
        )
    
    # Entrypoint
    workflow.set_entry_point("generate_questions")

    # Build the graph
    app = workflow.compile()
    return app

def short_node_schema(llm: ChatGroq | ChatOllama | ChatOpenAI):
    workflow = StateGraph(OneShotGraph)

    workflow.add_node("one_shot_node", lambda state: one_shot_node(state, llm))
    workflow.add_edge("one_shot_node", END)

    workflow.set_entry_point("one_shot_node")

    app = workflow.compile()
    return app