import os 
from apps.movie_demo_examples import examples
from langchain_community.graphs import Neo4jGraph


NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j+s://demo.neo4jlabs.com")
NEO4J_USER = os.environ.get("NEO4J_USER", "recommendations")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "recommendations")
NEO4J_DATABASE = "recommendations"

NEO4J_AUTH = (NEO4J_USER, NEO4J_PASS)


def get_schema():
    """
    Return the schema associated with the neo4j database.
    """
    graph = Neo4jGraph(url=NEO4J_URI, 
                    username=NEO4J_AUTH[0], 
                    password=NEO4J_AUTH[1],
                    database=NEO4J_DATABASE)
    return graph.schema


def get_neo4j_examples():
    return examples


# config for the ASTRIA Text2Cypher HyLo Graph
movie_demo_text2cypher_app_config = {
    "neo4j_uri": NEO4J_URI,
    "neo4j_auth": NEO4J_AUTH,
    "neo4j_database": NEO4J_DATABASE,
    "get_neo4j_examples": get_neo4j_examples,
    "get_neo4j_schema": get_schema,
    # "model_base_url": "https://ollama.pods.tacc.develop.tapis.io",
    "model_base_url": "http://localhost:11434",
    # "hylograph": "text2query_sim_graph",
    # "state": "Text2QueryGraphState",
    "hylograph": "text2query_sim_graph_execute",
    "state": "Text2QueryGraphExecuteState",
    "desc": "Text2Cyph-Movie-Llama3.1-8B",
}


if __name__ == '__main__':
    graph = Neo4jGraph(url=NEO4J_URI, 
                    username=NEO4J_AUTH[0], 
                    password=NEO4J_AUTH[1],
                    database=NEO4J_DATABASE)
    print(graph.schema)
