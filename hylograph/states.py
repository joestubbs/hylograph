from typing import List
from typing_extensions import TypedDict


class State(TypedDict):
    """
    A base class for all state objects to descend from.    
    Attributes:
      - app_config: The configuration for the app. Should include app-specific
        parameters, such as database coordinates, model URL and version, etc. 

    """
    app_config: dict


# class Text2QueryGraphState(TypedDict):
class Text2QueryGraphState(State):
    """
    Represents the state used in a Text2Query graph.
    Attributes:
     - question: the original question
     - examples: list of examples to use for prompting
     - prompt: the prompt 
     - generated_cypher: the generated cypher from the model 
     - query_error: error associated with the generated cypher
    """
    # app_config: dict
    question: str
    examples: List[dict]
    prompt: object 
    generated_cypher: str
    query_error: str


class Text2QueryGraphExecuteState(State):
    question: str
    examples: List[dict]
    prompt: object 
    generated_cypher: str
    query_error: str
    error_from_neo4j: object
    result: object 
