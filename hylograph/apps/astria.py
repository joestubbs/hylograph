"""
Configuration and functions for the ASTIAGraph project.
"""
import os 
from apps.astria_examples import examples


# Neo4j Instance ----------------

# L=Default to local instance:
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "tapis4ever")

NEO4J_AUTH = (NEO4J_USER, NEO4J_PASS)


def get_schema():
    """
    Return the schema associated with the neo4j database.
    """
    # read in the minimal schema 
    with open('apps/astria-graph-schema-min-v1.txt', 'r') as f:
        return f.read()


def get_neo4j_examples():
    """
    Gets a list of examples for ASTIA Graph.
    """
    return examples


# config for the ASTRIA Text2Cypher HyLo Graph
astria_text2cypher_app_config = {
    
    "neo4j_uri": NEO4J_URI,
    "neo4j_auth": NEO4J_AUTH,
    "get_neo4j_examples": get_neo4j_examples,
    "get_neo4j_schema": get_schema,
    # "model_base_url": "https://ollama.pods.tacc.develop.tapis.io",
    "model_base_url": "http://localhost:11434"

}