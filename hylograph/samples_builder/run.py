"""
Command line client for the samplers builder.
"""

import click 
import json
import nltk
import os 
from pathlib import Path
import sys 
# Python functionalities to collect and save data samples
from utils.utilities import *
# Neo4j graph connector
from utils.neo4j_conn import *
# Functionalities to extract schema and data from the graph
from utils.neo4j_schema import *
# Functionalities to parse extracted graph data
from utils.graph_utils import *


# Choose if to include repeats in the data builder or not
ALLOW_REPEATS = True

# Select the maximum size of each individual sampler with len(sampler) elements
M = 3


def get_graph_and_gutils(neo4j_url, neo4j_user, neo4j_pass, neo4j_database):
    # Initialize the Neo4j connector module
    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_pass, database=neo4j_database)

    # Module to extract data from the graph
    gutils = Neo4jSchema(url=neo4j_url, username=neo4j_user, password=neo4j_pass, database=neo4j_database)

    return graph, gutils



def gather_from_neo(neo4j_url, neo4j_user, neo4j_pass, neo4j_database, 
                    node_instances_size, rels_instances_size, base_output_path):
    """
    Initial step to gather schema and data from Neo4j and write to local files.
    """
    _, gutils = get_graph_and_gutils(neo4j_url, neo4j_user, neo4j_pass, neo4j_database)

    # Build the schema as a dictionary
    jschema = gutils.get_structured_schema
    
    # Check the output format
    jschema.keys()
    print(f"Got {len(jschema.keys())} total keys in the JSON schema.")
        
    # Extract the list of nodes
    nodes = get_nodes_list(jschema)
    
    # Extract the list of relationships
    relationships = jschema['relationships']
        
    # Extract the node instances from the graph
    node_instances = gutils.extract_node_instances(nodes, node_instances_size)

    # Extract the relationship instances from the graph
    rels_instances = gutils.extract_multiple_relationships_instances(relationships, rels_instances_size)

    # Serialize extracted neo4j.time data - for saving to json files
    nodes_instances_serialized = serialize_nodes_data(node_instances)
    rels_instances_serialized = serialize_relationships_data(rels_instances)
        
    # Save data to json files
    data_path = os.path.join(base_output_path, neo4j_database)
    
    # Ensure the data path directory exists
    Path(data_path).mkdir(parents=True, exist_ok=True)

    # Actually write the files
    print(f"Writing data to JSON files at path: {data_path}")
    write_json(jschema, os.path.join(data_path, "schema.json"))
    write_json(nodes_instances_serialized, os.path.join(data_path, "node_instances.json"))
    write_json(rels_instances_serialized, os.path.join(data_path, "relationship_instances.json"))


def get_nodes_props_instances_from_files(base_output_path, neo4j_database):
    
    data_path = os.path.join(base_output_path, neo4j_database)

    global jschema
    jschema = read_json(os.path.join(data_path, "schema.json"))
    
    global node_instances
    node_instances = read_json(os.path.join(data_path, "node_instances.json")) # these are serialized, see above
    
    global rels_instances
    rels_instances = read_json(os.path.join(data_path, "relationship_instances.json")) # these are serialized, see above
        
    # List of node labels
    global nodes 
    nodes = get_nodes_list(jschema)

    # Read the nodes with their properties and datatypes
    global node_props_types
    node_props_types = jschema['node_props']

    # Read the relationship properties with their datatypes
    global rel_props_types
    rel_props_types = jschema['rel_props']

    # Read the relationships as a list of triples
    global relationships
    relationships = jschema['relationships']
        
    # List of datatypes available as node properties in the graph
    global node_dtypes
    node_dtypes = retrieve_datatypes(jschema, "node")
    print(f"The datatypes for node properties: {node_dtypes}")
        
    # List of datatypes available as relationship properties in the graph
    global rel_dtypes
    rel_dtypes = retrieve_datatypes(jschema, "rel")
    print(f"The datatypes for relationships properties: {rel_dtypes}")

    # Extract and parse n instances of specified datatype, return a flatten list
    global dparsed
    dparsed = {f"{datatype.lower()}_parsed": \
                            parse_node_instances_datatype(jschema,
                                                        node_instances,
                                                        nodes,
                                                        datatype,
                                                        True) for datatype in node_dtypes
                                }
    # Add all the combined records
    dparsed['dtypes_parsed'] = sum(dparsed.values(), [])

    # Display available lists of instances
    print(f"A dictionary is created, the keys are: {dparsed.keys()}.")

    # Display a sample entry - instance of node and property with datatype STRING
    # BUG: This index of 11 was out of range for a previous run. 
    # print(f"Sample entry: {dparsed['string_parsed'][11]}.")
    print(f"Sample entry: {dparsed['string_parsed'][2]}.")

    # Generate all possible pairs of node properties datatypes
    global dtypes_pairs
    dtypes_pairs = list(product(node_dtypes, repeat=2))

    # Use dictionary comprehension with formatted keys for pairs
    global drels
    drels = {
        f"{dt1.lower()}_{dt2.lower()}_rels": \
        filter_relationships_instances(jschema, rels_instances, dt1, dt2)
        for dt1, dt2 in dtypes_pairs
    }

    # Add 'all_rels' key with concatenated lists from the other values
    drels['all_rels'] = sum(drels.values(), [])

    # Retain all those combinations that have nonempty entries
    drels = {key: value for key, value in drels.items() if value}

    # Display the list of node properties datatypes combinations for the relationships in the graph
    print(f"The possible end node properties datatypes pairs for relationships are\n {drels.keys()}.\n")

    # Sample entry
    print("A sample entry:")
    drels['string_string_rels'][1]    

    # Retrieve those instances for which relations have properties attached
    global instances_with_rel_props
    instances_with_rel_props = retrieve_instances_with_relationships_props(rels_instances)
    print(f"There are {len(instances_with_rel_props)} relationship(s) with properties in the graph.")

    # Display a sample
    instances_with_rel_props[0][0]
        
    # Use dictionary comprehension with formatted keys for pairs
    global drelsprops
    drelsprops = {
        f"{dt1.lower()}_{rt.lower()}_{dt2.lower()}_rels": \
        filter_relationships_with_props_instances(jschema, instances_with_rel_props, dt1, rt, dt2)
        for dt1, dt2 in dtypes_pairs
        for rt in rel_dtypes
        if filter_relationships_with_props_instances(jschema, instances_with_rel_props, dt1, rt, dt2)
    }

    # Add 'all_rels' key with concatenated lists from the other values
    drelsprops['all_rels'] = sum(drelsprops.values(), [])

    # Available combinations for source-relationship-target property datatypes
    print(f"The available combinations are {list(drelsprops.keys())}.")

    # Sample entry
    k = list(drelsprops.keys())[0]
    drelsprops[k][0]


# --------------------------------------------
# Cypher Builders

# Functionalities to parse extracted graph data
system_message =  "Convert the following question into a Cypher query using the provided graph schema!"


def count_nodes_of_given_label(neo4j_database):
    """ Determine how many nodes of specified label are in the graph."""

    def prompter(*params, **kwargs):

        label_1 = params[0]

        subschema =  build_minimal_subschema(jschema,[[label_1, ]],[], False, False, False)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the total number of {label_1} in the graph!""",
                   "NLQ": f"How many {label_1}s are there in the {neo4j_database} graph?",
                   "Complexity": "0,0",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) RETURN count(n)"
                   }
        return message

    sampler = []
    for label in nodes:
        temp_dict = prompter(label)
        sampler.append(temp_dict)

    return sampler


def match_one_node_one_prop(neo4j_database):
    """Return a given node label and a specified property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        # Extract subschema for the variables of interest
        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment

        # check the part of speech of prop_1 using NLTK; 
        pos = nltk.pos_tag([prop_1])[0][1]
        # if prop_1 is an adjective, any kind of verb (past, present, etc.) or an abvert
        if pos in ['JJ', 'VB', 'VBD', 'VBG', 'VBN', 'RN']:
            nlq = f"For each {label_1} in the {neo4j_database} graph, return when the {label_1} was {prop_1}."
        else:
            nlq = f"What is the {prop_1} of all {label_1}s in the {neo4j_database} graph?"
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch the {label_1} nodes and extract their {prop_1} property!""",
                   "NLQ": nlq,
                   "Complexity": "0,0",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) RETURN n.{prop_1}"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats=ALLOW_REPEATS)


def find_node_by_property(neo4j_database):
    """Find instances of given node label that has a property with specified value."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        # Extract subschema for the variables of interest
        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        
        # check the part of speech of prop_1 using NLTK; 
        pos = nltk.pos_tag([prop_1])[0][1]
        # if prop_1 is an adjective, any kind of verb (past, present, etc.) or an abvert
        if pos in ['JJ', 'VB', 'VBD', 'VBG', 'VBN', 'RN']:
            nlq = f"What {label_1} was {prop_1} in {val_1}?"
        else:
            nlq = f"What {label_1} has a {prop_1} of {val_1}?"
        
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {label_1} for which {prop_1} is {val_1}!""",
                   "NLQ": nlq,
                   "Complexity": "0,0",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1} {{{prop_1}:'{val_1}'}}) RETURN n"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)

# END of Samplers ---------------------------------------


def generate_samples(base_output_path, neo4j_database):
    """
    Actually call the individual samplers to generate the samples.
    This function can currently writes two identical files (but see which code block is commented out), 
      * parametric_trainer_with_repeats.json
      * generated_samples_all_fields.json
    Both of these files contain Prompt, Question, NLQ, Complexity, Schema, and Cypher 
    for every sample.
    """
    samplers = [count_nodes_of_given_label, match_one_node_one_prop, find_node_by_property]

    # List to collect the samples
    trainer=[]

    for s in samplers:
        sampler = s(neo4j_database)
        trainer += collect_samples(sampler, M)

    # Display the number of samples created and save the data to a file
    print(f"There are {len(trainer)} samples in the fine-tuning dataset.")

    data_path = os.path.join(base_output_path, neo4j_database)
    outpath = os.path.join(data_path, "generated_samples_all_fields.json")
    print(f"\nWriting samples to path: {outpath}")
    write_json(trainer, outpath)


def create_min_benchmark_samples_file(read_path, write_path):
    """
    Reads the benchmark file and creates a minimal samples JSON file.
    """
    
    with open(read_path, 'r') as f: 
        data = json.load(f)
    result = []
    for d in data:
        result.append({"question": d['NLQ'], "query": d["Cypher"], "complexity": d["Complexity"]})
    with open(write_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"A total of {len(result)} benchmark samples written to min benchmark file at path {write_path}")


def generate_files_for_params(neo4j_url, neo4j_user, neo4j_pass, neo4j_database, node_instances_size, 
                              rels_instances_size, base_output_path):
    """
    A parameterized version of main that can be passed parameters for the graph. 
    """
    gather_from_neo(neo4j_url, neo4j_user, neo4j_pass, neo4j_database, 
                    node_instances_size, rels_instances_size, base_output_path)
    get_nodes_props_instances_from_files(base_output_path, neo4j_database)
    generate_samples(base_output_path, neo4j_database)
    data_path = os.path.join(base_output_path, neo4j_database)
    read_path = os.path.join(data_path, "generated_samples_all_fields.json")
    write_path = os.path.join(data_path, "min_samples_benchmark.json")
    create_min_benchmark_samples_file(read_path, write_path)


# CLI Functions ------------------------------------------
@click.command
def install():
    """
    Installs the NLTK toolkit required for some functions.
    """
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')


@click.option('--conf', help='Path to JSON config file')
@click.command
def benchmark(conf):
    """
    Generate a benchmark dataset for one or more Neo4j instances. 
    Assumes a JSON config file describing the databases to use 
    """
    with open(conf, 'r') as f:
        data = json.load(f)
    try:
        base_output_path = data["base_output_path"]
    except KeyError:
        sys.exit("base_output_path is required.")
    try:
        node_instances_size = data['node_instances_size']
    except KeyError:
        sys.exit("node_instances_size is required.")
    try:
        rels_instances_size = data['rels_instances_size']
    except KeyError:
        sys.exit("rels_instances_size is required.")

    try:
        dbs = data['databases']
    except KeyError:
        sys.exit("databases is required.")
    for db in dbs:
        neo4j_url = db['neo4j_url']
        neo4j_user = db['neo4j_user']
        neo4j_pass = db['neo4j_pass']
        neo4j_database = db['neo4j_database']
        generate_files_for_params(neo4j_url, neo4j_user, neo4j_pass, neo4j_database, 
                        node_instances_size, rels_instances_size, base_output_path)


@click.group()
def cli():
    pass

cli.add_command(benchmark)
cli.add_command(install)


if __name__ == "__main__":
    cli()
