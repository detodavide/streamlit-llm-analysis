from langgraph.graph import StateGraph, END
from llm.langgraph.csv_user_question.states import GraphState as CPSGraphState
from llm.langgraph.csv_user_question.nodes import *
from llm.langgraph.csv_user_question.conditional_edges import *


def schema_builder():
    workflow = StateGraph(CPSGraphState)

    # Nodes
    workflow.add_node("get_answer", get_answer)
    workflow.add_node("rewrite_answer", rewrite_answer)
    workflow.add_node("web_search", web_search)

    # Edges
    workflow.add_edge("web_search", "rewrite_answer")
    workflow.add_edge("rewrite_answer", END)
    workflow.add_conditional_edges(
        "get_answer",
        route_to_research,
        {
            "research_info": "web_search",
            "rewrite_answer": "rewrite_answer",
        },
    )

    # Entrypoint
    workflow.set_entry_point("get_answer")

    # Build the graph
    app = workflow.compile()
    return app