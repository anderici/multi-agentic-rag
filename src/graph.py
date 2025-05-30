from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents import librarian_agent, editor_agent

class RagState(TypedDict):
    kb: str
    question: str
    chunks: str
    answer: str

def build_rag_graph():
    builder = StateGraph(RagState)  # Passa a classe TypedDict como schema
    
    builder.add_node("librarian", librarian_agent)
    builder.add_node("editor", editor_agent)
    
    builder.set_entry_point("librarian")
    builder.add_edge("librarian", "editor")
    builder.set_finish_point("editor")
    
    return builder.compile()
