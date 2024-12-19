from functools import partial
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from nodes import *
from edges import *
from states import Text2QueryGraphState, Text2QueryGraphExecuteState


# a basic text2query graph
text2query_graph_builder = StateGraph(Text2QueryGraphState)

# Add nodes
text2query_graph_builder.add_node("generate_examples", node_filter_neo4j_examples)
text2query_graph_builder.add_node("generate_cypher", node_generate_cypher)
text2query_graph_builder.add_node("validate_cypher", node_validate_cypher)
text2query_graph_builder.add_node("repair_cypher", node_repair_cypher)

# Add edges 
edge_query_has_error = partial(partial_edge_query_has_error, END)
text2query_graph_builder.add_edge(START, "generate_examples")
text2query_graph_builder.add_edge("generate_examples", "generate_cypher")
text2query_graph_builder.add_edge("generate_cypher", "validate_cypher")
text2query_graph_builder.add_conditional_edges("validate_cypher", 
                                               edge_query_has_error,
                                               {"repair_cypher": "repair_cypher",
                                                END: END})


# a text2query graph that includes a similarity checker node 
text2query_similarity_graph_builder = StateGraph(Text2QueryGraphState)

# Add nodes
text2query_similarity_graph_builder.add_node("get_question_similarity", node_get_question_similarity)
text2query_similarity_graph_builder.add_node("generate_examples", node_filter_neo4j_examples)
text2query_similarity_graph_builder.add_node("generate_cypher", node_generate_cypher)
text2query_similarity_graph_builder.add_node("validate_cypher", node_validate_cypher)
text2query_similarity_graph_builder.add_node("repair_cypher", node_repair_cypher)

# Add edges 
edge_query_has_error = partial(partial_edge_query_has_error, END)
text2query_similarity_graph_builder.add_edge(START, "get_question_similarity")
text2query_similarity_graph_builder.add_conditional_edges("get_question_similarity", 
                                                          edge_has_query,
                                                          {"generate_examples": "generate_examples",
                                                           "validate_cypher": "validate_cypher"})
text2query_similarity_graph_builder.add_edge("generate_examples", "generate_cypher")
text2query_similarity_graph_builder.add_edge("generate_cypher", "validate_cypher")
text2query_similarity_graph_builder.add_conditional_edges("validate_cypher", 
                                    edge_query_has_error,
                                    {"repair_cypher": "repair_cypher",
                                     END: END})


# a text2query graph that includes a similarity checker node and a node to actually 
# execute the cypher. 
text2query_sim_graph_execute_builder = StateGraph(Text2QueryGraphExecuteState)

# add nodes
text2query_sim_graph_execute_builder.add_node("get_question_similarity", node_get_question_similarity)
text2query_sim_graph_execute_builder.add_node("generate_examples", node_filter_neo4j_examples)
text2query_sim_graph_execute_builder.add_node("generate_cypher", node_generate_cypher)
text2query_sim_graph_execute_builder.add_node("validate_cypher", node_validate_cypher)
text2query_sim_graph_execute_builder.add_node("repair_cypher", node_repair_cypher)
text2query_sim_graph_execute_builder.add_node("execute_cypher", node_execute_cypher)

# Add edges 
edge_query_has_error = partial(partial_edge_query_has_error, "execute_cypher")
text2query_sim_graph_execute_builder.add_edge(START, "get_question_similarity")
text2query_sim_graph_execute_builder.add_conditional_edges("get_question_similarity", 
                                                          edge_has_query,
                                                          {"generate_examples": "generate_examples",
                                                           "validate_cypher": "validate_cypher"})
text2query_sim_graph_execute_builder.add_edge("generate_examples", "generate_cypher")
text2query_sim_graph_execute_builder.add_edge("generate_cypher", "validate_cypher")

text2query_sim_graph_execute_builder.add_conditional_edges("validate_cypher", 
                                    edge_query_has_error,
                                    {"repair_cypher": "repair_cypher",
                                     "execute_cypher": "execute_cypher"})
text2query_sim_graph_execute_builder.add_edge("execute_cypher", END)


# Remember results of graph executions within the same thread
memory = MemorySaver()

# compile the graph 
# TODO -- currently there is an issue using the checkpointer since it cannot serialize
#    function objects; it appears this is caused by the use of functions in the `state` 
#    object (e.g., the get_schema)
# text2query_graph = text2query_graph_builder.compile(checkpointer=memory)
text2query_graph = text2query_graph_builder.compile()

text2query_sim_graph = text2query_similarity_graph_builder.compile()

text2query_sim_graph_execute = text2query_sim_graph_execute_builder.compile()