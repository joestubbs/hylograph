"""
Entrypoint into the hylograph system.
"""
import logging 
from logs import get_logger
logger = get_logger("run.py", level=logging.INFO)


from graphs import text2query_graph, text2query_sim_graph
from states import Text2QueryGraphState
from apps.astria import astria_text2cypher_app_config


# config for the graph -- the thread id should be per person in a multi-user application
config = {"configurable": {"thread_id": "1"}}


# todo -- create a configurable system for these 
# graph = text2query_graph
graph = text2query_sim_graph


def get_new_state(question):
    state = Text2QueryGraphState()
    app_config = astria_text2cypher_app_config
    state["app_config"] = app_config
    state["question"] = question 
    return state 


def process_user_input(user_input: str):
    """
    Process a single input from the user by invoking the graph.
    """
    state = get_new_state(question=user_input)
    result = graph.invoke(state, config=config)
    logger.debug(f"Got result: {result}")
    query = result['generated_cypher']
    print(f"Assistant: Here is the query: {query}")


def chatbot():
    """
    Main loop for our chatbot. Processes user input and then feeds it to the graph.
    """
    logger.info("Starting chatbot")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        process_user_input(user_input)

    # while True:
    #     try:
    #         user_input = input("User: ")
    #         if user_input.lower() in ["quit", "exit", "q"]:
    #             print("Goodbye!")
    #             break
    #         process_user_input(user_input)
    #     except Exception as e:
    #         print(f"Got an exception trying to process the input. If you would like to quit, type 'quit'. \
    #               Details: {e}")
    #         break


def main():
    chatbot()


if __name__ == "__main__":
    main()
