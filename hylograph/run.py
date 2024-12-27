"""
Entrypoint into the hylograph system.
"""
import json
import logging 
from logs import get_logger
logger = get_logger("run.py", level=logging.INFO)
import functools
from langchain_community.graphs import Neo4jGraph

import click 
from enum import Enum
from graphs import text2query_graph, text2query_sim_graph, text2query_sim_graph_execute
from states import Text2QueryGraphState, Text2QueryGraphExecuteState
import sys 

from apps.astria import astria_text2cypher_app_config
from apps.movie_demo import movie_demo_text2cypher_app_config
from eval import evaluation


# config for the graph -- the thread id should be per person in a multi-user application
langraph_config = {"configurable": {"thread_id": "1"}}


class Apps(Enum):
    astria = "astria"
    demo = "demo"


def get_app_config(app: str):
    """
    Get the config associated with the app name
    """
    if app == "demo":
        app_config = movie_demo_text2cypher_app_config
    elif app == "astria":
        app_config = astria_text2cypher_app_config
    else:
        # if they enter an unrecognized app, quit (should be impossible)
        print("Invalid app value.")
        sys.exit(1)
    return app_config


def get_new_state(app_config, question):
    """
    Return a state object based on the app config.
    """
    # first, determine the class to use based on the `state` attribute of the config:
    if app_config['state'] == 'Text2QueryGraphExecuteState':
        state = Text2QueryGraphExecuteState()
    elif app_config['state'] == 'Text2QueryGraphState':
        state = Text2QueryGraphState()
    else:
        logger.error(f"Unrecognized state config {app_config['state']}; exiting...")
        sys.exit(1)
    state["app_config"] = app_config
    state["question"] = question 
    return state 


def get_hylograph(app_config):
    """
    Return the hylograph object associated with the config.
    """
    
    # For now, we base the graph on the hylograph config
    if app_config['hylograph'] == 'text2query_graph':
        graph = text2query_graph
    elif app_config['hylograph'] == 'text2query_sim_graph':
        graph = text2query_sim_graph
    elif app_config['hylograph'] == 'text2query_sim_graph_execute':
        graph = text2query_sim_graph_execute
    else:
        logger.error(f"Unrecognized hylograph config {app_config['hylograph']}; exiting...")
        sys.exit(1)
    return graph 


def process_benchmark_input(app_config, benchmark_input: str):
    """
    Process a single input from a benchmark by invoking the graph. Only the 
    exact query is returned.
    """
    state = get_new_state(app_config, question=benchmark_input)
    graph = get_hylograph(app_config)
    rsp = graph.invoke(state, config=langraph_config)
    return rsp['generated_cypher']


def process_user_input(user_input: str, app_config):
    """
    Process a single input from the user by invoking the graph. Analyzes the returned
    output to produce a sensible response to the end user, including response prompts, error handling
    and the included query used, if applicable. 
    """
    state = get_new_state(app_config, question=user_input)
    graph = get_hylograph(app_config)
    rsp = graph.invoke(state, config=langraph_config)
    logger.debug(f"type(rsp): {type(rsp)}")
    query = rsp['generated_cypher']
    
    # build the response; response depends on the graph and states though
    response = "Assistant: "
    # for graphs that execute the query, we look for errors from Neo4j as well as a final `result` 
    # attribute. 
    if app_config['state'] == "Text2QueryGraphExecuteState":
        error_from_neo4j = rsp.get('error_from_neo4j')
        result = rsp.get("result")
        if error_from_neo4j:
            response += f"We executed the query but got an error; details: {error_from_neo4j}"
        
        elif result:
            response += f"Answer: {result}"
        else:
            logger.info(f"Result was none, result: {result}")
            response += "We could not find the answer to that question."
                
        response += f"\nAssistant: Here is the query that was used: {query}"
    
    # otherwise, we just look for the query
    else:
        if query:
            response += f"Generated cypher query: {query}"
        else:
            query_error = rsp.get('query_error')
            response += f"Our attempts to generate cypher resulted in errors: {query_error}"

    return response


def get_app_graph(app_conf):
    """
    Return a Neo4jGraph object based on the app config. 
    """
    neo4j_uri = app_conf["neo4j_uri"]
    neo4j_auth = app_conf["neo4j_auth"]
    neo4j_database = app_conf.get("neo4j_database")
    if neo4j_database:
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_auth[0], password=neo4j_auth[1], database=neo4j_database)
    else:
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_auth[0], password=neo4j_auth[1])
    return graph


@click.option('--app', type=click.Choice([a.value for a in Apps]), default=Apps.demo.value, help='The app to execute.')
@click.command
def chatbot(app):
    """
    Main loop for our chatbot. Processes user input and then feeds it to the graph.
    """
    logger.info("Starting chatbot")

    app_config = get_app_config(app)
    while True:
        user_input = click.prompt(click.style("\nUser: ", fg="green"))
        if user_input.lower() in ["quit", "exit", "q"]:
            click.echo(click.style("Goodbye!", fg="red"))
            break
        
        reply = process_user_input(user_input, app_config)
        click.echo(click.style(reply, fg="blue"))


@click.option('--app', type=click.Choice([a.value for a in Apps]), default=Apps.demo.value, help='The app to execute.')
@click.option('--benchmark', help='Path to the benchmark JSON file to use.')
@click.option('--outfile', help='Path to write the output report to.', default="output.csv")
@click.command
def benchmark(app, benchmark, outfile):
    """
    Run a benchmark test against a graph.
    """
    logger.info(f"Starting benchmark for app {app} and file {benchmark}")
    app_config = get_app_config(app)
    model_app_desc = app_config["desc"]
    logger.info(f"Got config, running benchmark for: {model_app_desc}")
    graph = get_app_graph(app_config)

    with open(benchmark, 'r') as f:
        data = json.load(f)
    
    # Define a partial with the app specified since the `invoke_model` parameter to evaluation
    # doesn't know about apps. 
    invoke_model = functools.partial(process_benchmark_input, app_config)
    
    # The evaluation function iterates over the items in the benchmark
    evaluation(benchmark=data, 
               invoke_model=invoke_model, 
               graph=graph, 
               model_desc=model_app_desc, 
               out_file=outfile)


@click.group()
def cli():
    pass

    
cli.add_command(chatbot)
cli.add_command(benchmark)


if __name__ == "__main__":
    cli()
