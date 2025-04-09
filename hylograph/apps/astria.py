"""
Configuration and functions for the ASTIAGraph project.
"""
import os 
from apps.astria_examples import examples


OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
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
    "model_base_url": "http://localhost:11434",
    "hylograph_chromadb_path": "/tmp/hylograph/astria/chroma",
    "sqlitecache_path":"/tmp/langchain.db",
    "hylograph": "text2query_sim_graph",
    "state": "Text2QueryGraphState",
    # "hylograph": "text2query_sim_graph_execute",
    # "state": "Text2QueryGraphExecuteState",
    "desc": "Text2Cyph-Astria-Llama3.1-8B",

}

benchmark = {
    "low_question_low_query": [
        # {
        #     "question": "",
        #     "query": "",
        #     "result": "",
        # },
        {
            "question": "How many sensors are in the catalog?",
            "query": "MATCH (o:Sensor) RETURN count(o) as nodes",
            "result": "6",
        },
        {
            "question": "How many measurements are in the catalog?",
            "query": "MATCH (o:Measurement) RETURN count(o) as nodes",
            "result": "259100",
        },
        {
            "question": "How many space objects in the catalog are from the US?",
            "query": "MATCH (o:SpaceObject {Country: 'US'}) RETURN count(o) as nodes",
            "result": "16067397",
        },
        {
            "question": "Which space object has the smallest AreaToMass?",
            "query": "MATCH (d:SpaceObject) RETURN d ORDER BY d.AreaToMass LIMIT 1", 
            "result": "(:SpaceObject:'2022-11-05' {AreaToMass: 3.94E-6})"
        },

    ],

    "low_question_high_query": [
        {
            "question": "Which data sources are not public?",
            "query": "MATCH (d: DataSource {PublicData: FALSE}) RETURN d.Name",
            "result": "",
        },

    ],

    "high_question_low_query": [
        # {
        #     "question": "",
        #     "query": "",
        #     "result": "",
        # },

    ],

    "high_question_high_query": [
        {
            "question": "How many measurements were taken in the year 2020 of the  on ",
            "query": "",
            "result": "",
        },
    ],

}
"MATCH (o:SpaceObject {Name: 'VANGUARD 1'})-- (n) RETURN count(n) AS connectedNodes"
"MATCH (o:SpaceObject {Name: 'VANGAURD 1'})-[r]-(b) RETURN r"
