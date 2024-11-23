from logs import get_logger
logger = get_logger("edges.py")

from langgraph.graph import END
from states import Text2QueryGraphState


# Edges -----

def edge_query_has_error(state: Text2QueryGraphState):
    """
    LangGraph edge to check whether a query has an error, and if so, route to the repair node,
    otherwise, route to the END.

    Required Configuration: None

    Required State:
     - query_error: string containing the error, or None.
   
    Modifies State: None 

    """
    logger.info("Executing edge_query_has_error")
    if state["query_error"]:
        logger.info("edge_query_has_error completed, returning repair_cypher")
        return "repair_cypher"
    logger.info("edge_query_has_error completed, returning END")
    return END 
