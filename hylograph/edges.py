from logs import get_logger
logger = get_logger("edges.py")

from langgraph.graph import END
from states import Text2QueryGraphState


# Edges -----

def partial_edge_query_has_error(next_state, state: Text2QueryGraphState):
    """
    A partial LangGraph edge to check whether a query has an error, and if so, route to the repair node,
    otherwise, route to the next_state. This function should be compiled using partial by providing the
    required compile time args.

    Required compile time: next_state: either END or a string representing the next state to transition
    to in case the query does not contain an error. 

    Required Configuration: None

    Required State:
     - query_error: string containing the error, or None.
   
    Modifies State: None 

    """
    logger.info("Executing edge_query_has_error")
    if state["query_error"]:
        logger.info("edge_query_has_error completed, returning repair_cypher")
        return "repair_cypher"
    logger.info(f"edge_query_has_error completed; returning {next_state}")
    return next_state 


def edge_has_query(state: Text2QueryGraphState):
    """
    LangGraph edge to check whether a query has been generated, and if so, route to the validation
    edge; otherwise, route to the generate_examples node.
    """
    logger.info("Executing edge_has_query")
    if state.get("generated_cypher"):
        logger.info("cypher was generated, passing to validator")
        return "validate_cypher"
    # Ideally, this would be dynamic somehow; i.e., would support "the next" node, whatever that might be
    # Here, we have to have "global" knowledge of the graph.
    logger.info(f"No generated_cypher in state;") #current state: {state}")
    return "generate_examples"