from langgraph.graph import StateGraph, END
from llm.langgraph.csv_point_selection.states import GraphState as CPSGraphState
from llm.langgraph.csv_point_selection.nodes import *

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