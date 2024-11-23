import logging
from neo4j import GraphDatabase
import os 

from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from logs import get_logger
from states import Text2QueryGraphState


logger = get_logger(name="nodes.py", level=logging.INFO, strategy="stream")

# These next three lines swap the stdlib sqlite3 lib with the pysqlite3 package
# This is required by the Chroma package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Defaults
DEFAULT_MODEL_URL = "https://ollama.pods.tacc.develop.tapis.io"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_LLM = "llama3.1:8b"


# Implementation Function

def validate_cypher(query, neo4j_uri, neo4j_auth): 
    """
    Given a `query`, returns None if the query is valid, otherwise, returns the error associated 
    with the query, as a string.
    """
    with GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) as driver:
        with driver.session() as session:
            try:
                session.run("EXPLAIN " + query)
                return None
            except Exception as e:
                if hasattr(e, "message"):
                    message = e.message 
                else:
                    message = f"exception: {e}"
                return message


def _get_embedding_model(model=DEFAULT_EMBEDDING_MODEL, 
                         base_url=DEFAULT_MODEL_URL):
    """
    Returns the embedding model used for this graph.
    """
    return OllamaEmbeddings(model=model, base_url=base_url)


def get_example_selector(all_examples, 
                         nbr_examples=5, 
                         model=DEFAULT_EMBEDDING_MODEL, 
                         base_url=DEFAULT_MODEL_URL):
    """
    Returns the example selector that is used to select a subset of `all_examples` that are the most
    similar to the question.
    """
    model = _get_embedding_model(model, base_url)
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples = all_examples,
        embeddings = model,
        # vectorstore_cls = Chroma(persist_directory="/home/jstubbs/tmp/ASTRIA/chroma"),
        vectorstore_cls = Chroma,
        k=nbr_examples,   
    )
    return example_selector


def filter_examples(question, 
                    all_examples, 
                    nbr_examples=5,
                    model=DEFAULT_EMBEDDING_MODEL, 
                    base_url=DEFAULT_MODEL_URL):
    """
    Return the most relevant `nbr_examples` to the `question`.
    """
    example_selector = get_example_selector(all_examples, nbr_examples, model, base_url)
    return example_selector.select_examples({"question": question})


def _get_llm(model=DEFAULT_LLM, base_url=DEFAULT_MODEL_URL):
    """
    Returns the LLM for this graph.
    """
    return ChatOllama(model=model, base_url=base_url)


def create_few_shot_cypher_prompt(examples):
    """
    Create a few short prompt template to instruct an LLM to generate a Cypher query 
    without context variable. 
    
     - examples: a list of dictionaries of cypher examples; each dict should have a `question` and
                 `answer` key.
    """

    # First, create the PromptTemplate that formats the examples into a string
    example_prompt = PromptTemplate(
        input_variables=["question", "query"],
        template="Question: {question}\nCypher query: {query}"
    )

    # The prefix for the prompt.
    prefix = """
    Task:Generate Cypher statement to query a graph database.
    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.

    Note: Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    Do not include any text except the generated Cypher statement.

    Schema: {schema}

    Examples: Here are a few examples of generated Cypher statements for particular questions:
    """
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix="Question: {question}, \nCypher Query: ",
        input_variables =["question", "query"],
    ) 
    return few_shot_prompt


def create_cypher_repair_prompt():
    """
    Create a prompt template to instruct an LLM to repair a previously generated Cypher query.
    """
    template = """
    Task: Repair the previously generated cypher query to fix the listed errors.
    Instructions: The Cypher statement you generated to query a graph database was not correct. 
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.
    Here is the original question: {question}
    Here is the query you generated that had errors: {query}
    Here are the errors associated with the query you generated: {error}
    Cypher query: 
    """
    prompt = PromptTemplate(input_variables=["question", "query", "error"], template=template)
    return prompt


def invoke_llm_chain(prompt_template, template_args, model=DEFAULT_LLM, base_url=DEFAULT_MODEL_URL):
    """
    Calls the LLM with a prompt and returns the reply.
     - prompt_template: a ChatPromptTemplate created, for example, using ChatPromptTemplate.from_message()
     - template_args: dictionary of template variables and values. 

    """
    llm = _get_llm(model, base_url)
    parser = StrOutputParser()
    chain = prompt_template | llm | parser
    result = chain.invoke(template_args)
    return result


# Nodes -------

def node_filter_neo4j_examples(state: Text2QueryGraphState):
    """
    LangGraph node that uses an example_filter to generate relevant examples for prompting
    an LLM to generate a cypher query.

    Required Configuration: 
     - app_config["get_neo4j_examples"]: Callable returning the app's Neo4j examples.

    Optional Configuration: 
    - app_config["model_base_url"]: URL serving the models
    - app_config["embedding_model"]: Name of an embedding model to use, e.g.,

    Required State: None
   
    Modifies State:
     - examples: 

    """
    logger.info(f"Executing node_filter_neo4j_examples")
    app_config = state["app_config"]
    get_neo4j_examples = app_config["get_neo4j_examples"]
    
    # Optional configs
    model_base_url = DEFAULT_MODEL_URL
    if "model_base_url" in app_config.keys():
        model_base_url = app_config["model_base_url"]

    model = DEFAULT_EMBEDDING_MODEL
    if "embedding_model" in app_config.keys():
        model = app_config["embedding_model"]

    state["examples"] = filter_examples(all_examples=get_neo4j_examples(), 
                                        question=state["question"],
                                        model=model,
                                        base_url=model_base_url)
    
    logger.info("node_filter_neo4j_examples complete.")
    return state 


def node_generate_cypher(state: Text2QueryGraphState):
    """
    LangGraph node to generate a cypher query.

    Required Configuration: 
     - app_config["get_neo4j_schema"]: Callable returning the app's Neo4j schema.

    Required State:
     - question: A user-provided question
     - examples: A set of Neo4j examples to include in the prompt 

    Optional Configuration: 
    - app_config["model_base_url"]: URL serving the models
    - app_config["llm"]: Name of an embedding model to use, e.g.,
    
    Modifies State:
     - prompt: a few-shot learning prompt to generate a cypher query. 
     - generated_cypher: The cypher generated by the model to answer the question.

    Note: this node does not validate not execute the cypher query

    """
    logger.info(f"Executing node_generate_cypher")
    question = state["question"]
    examples = state["examples"]
    prompt = create_few_shot_cypher_prompt(examples=examples)
    state["prompt"] = prompt

    app_config = state["app_config"]
    get_schema = app_config["get_neo4j_schema"]
    template_args = {
            "schema": get_schema(),
            "question": question,
        }

    # Optional configs
    model_base_url = DEFAULT_MODEL_URL
    if "model_base_url" in app_config.keys():
        model_base_url = app_config["model_base_url"]

    model = DEFAULT_LLM
    if "llm" in app_config.keys():
        model = app_config["llm"]

    generated_cypher = invoke_llm_chain(prompt_template=prompt, 
                                        template_args=template_args,
                                        model=model,
                                        base_url=model_base_url
                                        )
    state["generated_cypher"] = generated_cypher
    logger.info(f"node_generate_cypher complete; cypher generated: {generated_cypher}")
    return state


def node_validate_cypher(state: Text2QueryGraphState):
    """
    LangGraph node to validate the generated cypher.

    Required Configuration: 
     - app_config["neo4j_URI"]: String representing the NEO4J_URI.
     - app_config["neo4j_AUTH"]: Tuple representing the NEO4J_AUTH.

    Required State:
     - generated_cypher: A cypher query previously generatd. 
    
    Modifies State:
     - query_error: A string representing the error associated with the query, or None, 
       if the query is valid. 

    """
    logger.info(f"Executing node_validate_cypher")
    query = state["generated_cypher"]
    app_config = state["app_config"]

    neo4j_uri = app_config["neo4j_uri"]
    neo4j_auth = app_config["neo4j_auth"]

    error = validate_cypher(query=query, neo4j_uri=neo4j_uri, neo4j_auth=neo4j_auth)    
    # error will either be None or a string representing the error
    state["query_error"] = error
    logger.info(f"node_validate_cypher complete; query_error: {error}")
    return state


def node_repair_cypher(state: Text2QueryGraphState):
    """
    LangGraph node to use the LLM to try and repair a previously generated cypher query 
    that has an error.
    """
    logger.info(f"Executing node_repair_cypher")
    question = state["question"]
    query = state["generated_cypher"]
    error = state["query_error"]
    prompt = create_cypher_repair_prompt()
    state["prompt"] = prompt
    template_args = {
            "question": question,
            "query": query,
            "error": error, 
        }
    app_config = state["app_config"]
    # Optional configs    
    model_base_url = DEFAULT_MODEL_URL
    if "model_base_url" in app_config.keys():
        model_base_url = app_config["model_base_url"]

    model = DEFAULT_LLM
    if "llm" in app_config.keys():
        model = app_config["llm"]
    
    generated_cypher = invoke_llm_chain(prompt_template=prompt, 
                                        template_args=template_args,
                                        model=model,
                                        base_url=model_base_url)
    state["generated_cypher"] = generated_cypher
    logger.info(f"node_repair_cypher complete; generated_cypher: {generated_cypher}")
    return state
