"""
Entrypoint into the hylograph system.
"""
import logging 
from logs import get_logger
logger = get_logger("run.py", level=logging.INFO)

import click 
from enum import Enum
from graphs import text2query_graph, text2query_sim_graph, text2query_sim_graph_execute
from states import Text2QueryGraphState, Text2QueryGraphExecuteState
import sys 
from apps.astria import astria_text2cypher_app_config
from apps.movie_demo import movie_demo_text2cypher_app_config


# config for the graph -- the thread id should be per person in a multi-user application
config = {"configurable": {"thread_id": "1"}}


# todo -- create a configurable system for these 
# graph = text2query_graph
# graph = text2query_sim_graph
graph = text2query_sim_graph_execute

class Apps(Enum):
    astria = "astria"
    demo = "demo"


def get_new_state(app, question):
    # state = Text2QueryGraphState()
    state = Text2QueryGraphExecuteState()
    if app == "demo":
        app_config = movie_demo_text2cypher_app_config
    elif app == "astria":
        app_config = astria_text2cypher_app_config
    else:
        # if they enter an unrecognized app, quit (should be impossible)
        print("Invalid app value.")
        sys.exit(1)
    state["app_config"] = app_config
    state["question"] = question 
    return state 


def process_user_input(user_input: str, app: str):
    """
    Process a single input from the user by invoking the graph.
    """
    state = get_new_state(app, question=user_input)
    result = graph.invoke(state, config=config)
    logger.info(f"type(result): {type(result)}")
    try:
        logger.info(f"Keys of result: {result.keys()}")
    except: pass 
    query = result['generated_cypher']
    error_from_neo4j = result.get('error_from_neo4j')
    records = result.get("records")
    
    # build the response
    response = "Assistant: "
    if error_from_neo4j:
        response += f"We executed the query but got an error; details: {error_from_neo4j}"
    
    elif records:
        try:
            tot_records = len(records)
        except Exception as e:
            logger.error(f"Error getting length of records: {records}; details: {e}")
            response += "\nWe failed to parse the result. See the logs"
            return response
        if tot_records < 1:
            logger.error(f"No records returned; records: {records}")
            response += "\nWe failed to obtain any result. See the logs."
            return response
        response += "Answer: "
        for r in records:
            try:
                response += f"{r.data()}\n"
            except Exception as e:
                logger.error(f"Error parring the result {r}; details: {e}")
    else:
        logger.info(f"Records was none, records:{records}")
        response += "We could not find the answer to that question."
            
    response += f"\nAssistant: Here is the query that was used: {query}"
    return response


@click.option('--app', type=click.Choice([a.value for a in Apps]), default=Apps.demo.value, help='The app to execute.')
@click.command
def chatbot(app):
    """
    Main loop for our chatbot. Processes user input and then feeds it to the graph.
    """
    logger.info("Starting chatbot")

    while True:
        user_input = click.prompt(click.style("User: ", fg="green"))
        if user_input.lower() in ["quit", "exit", "q"]:
            click.echo(click.style("Goodbye!", fg="red"))
            break
        reply = process_user_input(user_input, app)
        click.echo(click.style(reply, fg="blue"))


@click.command
def benchmark():
    """
    Run a benchmark test against a graph.
    """
    pass


@click.group()
def cli():
    pass

    
cli.add_command(chatbot)
cli.add_command(benchmark)


if __name__ == "__main__":
    cli()
