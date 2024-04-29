from langgraph.graph import StateGraph, END
from llm.langgraph.states.csv_point_selection_state import GraphState as CPSGraphState
from llm.langgraph.nodes.csv_point_selection_nodes import *

# Init Graph
def schema_builder():
    workflow = StateGraph(CPSGraphState)

    # Nodes
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("questions_answering", questions_answering)
    workflow.add_node("summarize_answers", summarize_answers)

    # Edges
    workflow.add_edge("generate_questions", "questions_answering")
    workflow.add_edge("questions_answering", "summarize_answers")
    workflow.add_edge("summarize_answers", END)

    # Entrypoint
    workflow.set_entry_point("generate_questions")

    # Build the graph
    app = workflow.compile()
    return app