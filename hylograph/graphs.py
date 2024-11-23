from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from nodes import *
from edges import *
from states import Text2QueryGraphState


text2query_graph_builder = StateGraph(Text2QueryGraphState)

# Add nodes
text2query_graph_builder.add_node("generate_examples", node_filter_neo4j_examples)
text2query_graph_builder.add_node("generate_cypher", node_generate_cypher)
text2query_graph_builder.add_node("validate_cypher", node_validate_cypher)
text2query_graph_builder.add_node("repair_cypher", node_repair_cypher)

# Add edges 
text2query_graph_builder.add_edge(START, "generate_examples")
text2query_graph_builder.add_edge("generate_examples", "generate_cypher")
text2query_graph_builder.add_edge("generate_cypher", "validate_cypher")
text2query_graph_builder.add_conditional_edges("validate_cypher", 
                                    edge_query_has_error,
                                    {"repair_cypher": "repair_cypher",
                                     END: END})


# Remember results of graph executions within the same thread
memory = MemorySaver()

# compile the graph 
# TODO -- currently there is an issue using the checkpointer since it cannot serialize
#    function objects; it appears this is caused by the use of functions in the `state` 
#    object (e.g., the get_schema)
# text2query_graph = text2query_graph_builder.compile(checkpointer=memory)
text2query_graph = text2query_graph_builder.compile()
