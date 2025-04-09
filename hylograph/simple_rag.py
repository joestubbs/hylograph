from langchain import hub
from langgraph.graph import START, END, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from vector import get_vector_store
from models import get_llm


# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    db = get_vector_store(collection="test")
    retrieved_docs = db.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    
    llm = get_llm()
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State)
graph_builder.add_node(retrieve)
graph_builder.add_node(generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()


if __name__ == "__main__":
    response = graph.invoke({"question": "What is a Tapis associate site?"})
    print(response["answer"])