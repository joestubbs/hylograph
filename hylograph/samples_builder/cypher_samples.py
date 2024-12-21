"""
Code to generate a samples of the form {"question": .., "query": ..} based only on a Neo4j schema.
The code in this file and the utils directory are based on the Neo4jLabs code in
the repository: https://github.com/neo4j-labs/text2cypher

Cf., for example, the jupyter notebook here:
 https://github.com/neo4j-labs/text2cypher/blob/main/datasets/functional_cypher/SFT_Functional_Data_Builder.ipynb

"""
import os 
import json 

# ToDo: Update these --- 
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j+s://demo.neo4jlabs.com")
NEO4J_USER = os.environ.get("NEO4J_USER", "recommendations")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "recommendations")
NEO4J_DATABASE = "recommendations"

NEO4J_AUTH = (NEO4J_USER, NEO4J_PASS)


# Python functionalities to collect and save data samples
from utils.utilities import *
# Neo4j graph connector
from utils.neo4j_conn import *
# Functionalities to extract schema and data from the graph
from utils.neo4j_schema import *
# Functionalities to parse extracted graph data
from utils.graph_utils import *
     

# Initialize the Neo4j connector module
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS, database=NEO4J_DATABASE)

# Module to extract data from the graph
gutils = Neo4jSchema(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS, database=NEO4J_DATABASE)
     
# Check connectivity to the db:
r = graph.query("MATCH (n) RETURN COUNT(n) AS TotalNodes")
print(r)


# ToDo: Configure directories --------------------------------

# Create a path variable for the data folder
data_path = '/home/jstubbs/tmp/cypher_samples/movie-demo/'

# Graph schema file
schema_file = 'schema_file.json'

# Node and relationships instances
node_instances_file = 'node_instances_file.json'
rels_instances_file = 'rels_instances_file.json'

# Fine-tuning datasets
trainer_with_repeats_file = 'parametric_trainer_with_repeats.json'
trainer_without_repeats_file = 'parametric_trainer_without_repeats.json'


# ToDo: Configure options for Building Samples --------------------------------
     
# Choose how many instances of each node label to extract
node_instances_size = 12

# Choose how many instances of each relationship type to extract
rels_instances_size = 12

# Choose if to include repeats in the data builder or not
ALLOW_REPEATS = True

# Select the maximum size of each individual sampler with len(sampler) elements
M = 50
     

# A list of all samplers, for convenience 
all_samplers = []
with open("all_samplers2.txt", 'r') as f:
    for s in f:
        all_samplers.append(s)


def gather_from_neo():
    """
    Initial step to gather schema and data from Neo4j and write to local files.
    """
    # Build the schema as a dictionary
    jschema = gutils.get_structured_schema
    # Check the output format
    jschema.keys()
        
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
    write_json(jschema, data_path+schema_file)
    write_json(nodes_instances_serialized, data_path+node_instances_file)
    write_json(rels_instances_serialized, data_path+rels_instances_file)
        

def get_nodes_props_instances_from_files():
    """
    Gathers initial datastructures from the files, written in the previous step. 
    """

    # Read the data from files if previously saved
    global jschema
    jschema = read_json(data_path+schema_file)
    
    global node_instances
    node_instances = read_json(data_path+node_instances_file) # these are serialized, see above
    
    global rels_instances
    rels_instances = read_json(data_path+rels_instances_file) # these are serialized, see above
        
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
    print(f"Sample entry: {dparsed['string_parsed'][11]}.")

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

def count_nodes_of_given_label():
    """ Determine how many nodes of specified label are in the graph."""

    def prompter(*params, **kwargs):

        label_1 = params[0]

        subschema =  build_minimal_subschema(jschema,[[label_1, ]],[], False, False, False)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the total number of {label_1} in the graph!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) RETURN count(n)"
                   }
        return message

    sampler = []
    for label in nodes:
        temp_dict = prompter(label)
        sampler.append(temp_dict)

    return sampler
     

def paths_with_node_endpoint():
    """Find paths with specified endpoints."""

    def prompter(*params, **kwargs):

        label_1 = params[0]

        subschema = build_minimal_subschema(jschema,[[label_1, ]],[], False, False, False)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Identify three paths where {label_1} is a start or end node!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH p=(b:{label_1})-[r*]->(n) RETURN p UNION MATCH p=(n)-[r*]->(b:{label_1}) RETURN p LIMIT 3"
                   }
        return message

    sampler = []

    for label in nodes:
        temp_dict = prompter(label)
        sampler.append(temp_dict)

    return sampler     


def match_one_node_one_prop():
    """Return a given node label and a specified property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        # Extract subschema for the variables of interest
        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment

        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch the {label_1} nodes and extract their {prop_1} property!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) RETURN n.{prop_1}"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats=ALLOW_REPEATS)


def where_one_node_one_prop_notnull_numeral():
    """Return n (use figures, e.g. 8) nodes where a property is not null."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find 10 {label_1} that have the {prop_1} recorded and return these values!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NOT NULL RETURN n.{prop_1} LIMIT 10"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats = ALLOW_REPEATS)


def where_one_node_one_prop_notnull_literal():
    """Return n (use words, e.g. eight) nodes where a property is not null."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find ten {label_1} that have {prop_1} and return their records!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NOT NULL RETURN n.{prop_1} LIMIT 10"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats = ALLOW_REPEATS)


def where_one_node_one_prop_null_numeral():
    """Return n (use figures, e.g. 8) nodes where a property is null."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find 8 {label_1} that are missing the {prop_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NULL RETURN n LIMIT 8"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)


def find_node_notproperty_count():
    """Find how many nodes of given label are missing a specified property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the total number of {label_1} for which the {prop_1} is missing!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NULL RETURN count(n)"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)


def find_node_property_count():
    """Count nodes of given label which have a certain property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the total number of {label_1} that have the {prop_1} recorded!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NOT NULL RETURN count(n)"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)


def find_node_by_property():
    """Find instances of given node label that has a property with specified value."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        # Extract subschema for the variables of interest
        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {label_1} for which {prop_1} is {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1} {{{prop_1}:'{val_1}'}}) RETURN n"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)

def match_skip_limit_return_property():
    """Return a list of values of a property, using skip and limit."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        nrecs = kwargs.get('nrecs', 2)

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Return the {prop_1} of the {label_1}, skip the first {nrecs} records and return {nrecs} records!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) RETURN n.{prop_1}  SKIP {nrecs} LIMIT {nrecs}"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)



def where_one_node_one_prop_notnull_numeral():
    """Return n (use figures, e.g. 8) nodes where a property is not null."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find 10 {label_1} that have the {prop_1} recorded and return these values!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NOT NULL RETURN n.{prop_1} LIMIT 10"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats = ALLOW_REPEATS)


def where_one_node_one_prop_notnull_literal():
    """Return n (use words, e.g. eight) nodes where a property is not null."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find ten {label_1} that have {prop_1} and return their records!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NOT NULL RETURN n.{prop_1} LIMIT 10"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats = ALLOW_REPEATS)


def where_one_node_one_prop_null_numeral():
    """Return n (use figures, e.g. 8) nodes where a property is null."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find 8 {label_1} that are missing the {prop_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NULL RETURN n LIMIT 8"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)


def find_node_notproperty_count():
    """Find how many nodes of given label are missing a specified property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the total number of {label_1} for which the {prop_1} is missing!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NULL RETURN count(n)"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)


def find_node_property_count():
    """Count nodes of given label which have a certain property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the total number of {label_1} that have the {prop_1} recorded!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NOT NULL RETURN count(n)"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)


def find_node_by_property():
    """Find instances of given node label that has a property with specified value."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        # Extract subschema for the variables of interest
        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {label_1} for which {prop_1} is {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1} {{{prop_1}:'{val_1}'}}) RETURN n"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def match_skip_limit_return_property():
    """Return a list of values of a property, using skip and limit."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        nrecs = kwargs.get('nrecs', 2)

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Return the {prop_1} of the {label_1}, skip the first {nrecs} records and return {nrecs} records!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) RETURN n.{prop_1}  SKIP {nrecs} LIMIT {nrecs}"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


# String Datatype ---

def match_where_skip_limit_return_property():
    """Fetch a list of nodes with certain properties, use skip and limit."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        nrecs = kwargs.get('nrecs', 2)

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {label_1} for which {prop_1} starts with {val_1[0]}, skip the first {nrecs} records and return the next {nrecs} records of {prop_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} STARTS WITH '{val_1[0]}' WITH n.{prop_1} AS {prop_1} SKIP {nrecs} LIMIT {nrecs} RETURN {prop_1}"
                   }
        return message

    return build_node_sampler(dparsed["string_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def where_one_node_one_prop_one_val():
    """Retrieve nodes of given label where a string property has a given value."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {label_1} where {prop_1} is {val_1.strip()}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} = '{val_1}' RETURN n"
                   }
        return message

    return build_node_sampler(dparsed["string_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def where_one_node_one_string_contains():
    """Retrieve nodes of specified label where a string property contains a given substring."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {label_1} where {prop_1} contains {val_1[:5]}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} CONTAINS '{val_1[:5]}' RETURN n"
                   }
        return message

    return build_node_sampler(dparsed["string_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def find_node_by_start_substring():
    """Find instances of given node label that has a property that starts with a specified substring."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {label_1} for which {prop_1} starts with {val_1[:3]}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} STARTS WITH '{val_1[:3]}' RETURN n"
                   }
        return message

    return build_node_sampler(dparsed["string_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def where_one_node_string_re():
    """Retrieve nodes of given label with a string property satisfies a condition given by a regular expression."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch the {label_1} where {prop_1} ends with {val_1[:2]}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} =~'{val_1[:2]}.*' RETURN n"
                   }
        return message

    return build_node_sampler(dparsed["string_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


# Temporal Data Types ---

def find_count_in_interval():
    """Node count for a given time interval."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""How many {label_1} have {prop_1} between January 1, 2010 and January 1, 2015?!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} >= date('2010-01-01') AND n.{prop_1} <= date('2015-01-01') RETURN count(n) AS {label_1}s"
        }
        return message


    return build_node_sampler(dparsed["date_parsed"], # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)


def find_nodes_today():
    """Find nodes with property dated within the last 24 hours."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""List {label_1} that have {prop_1} in the last 24 hours!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} > datetime() - duration('P1D') RETURN n"
        }
        return message

    return build_node_sampler(dparsed["date_parsed"],  # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)


def find_nodes_monday():
    """Find the count of nodes with given label and specified property dated on a Monday."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""How many {label_1} have {prop_1} on a Monday?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE date(n.{prop_1}).weekday = 1 RETURN count(n)"
        }
        return message

    return build_node_sampler(dparsed["date_parsed"],  # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)

def find_property_after_hour():
    """Find the count of nodes with given label and specified property dated after a given date and time."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find how many {label_1}s have {prop_1} after 6PM, January 1, 2020?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} >= datetime('2010-01-01T18:00:00') RETURN count(n) AS {label_1}s"
        }
        return message

    return build_node_sampler(dparsed["date_parsed"],  # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                                 prompter,
                                 allow_repeats= ALLOW_REPEATS)


def where_one_node_one_prop_equals_year():
    """Retrieve nodes of given label where a property has a specific year."""


    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        date_year = kwargs.get('date_year', 2010)

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch {label_1} where {prop_1} is in {date_year}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE date(n.{prop_1}).year = {date_year} RETURN n"
                   }
        return message

    return build_node_sampler(dparsed["date_parsed"],  # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def where_one_node_one_prop_equals_date():
    """Retrieve nodes of given label where a date property has a specified value."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find {label_1} such that {prop_1} is {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} = date('{val_1}') RETURN n"
                   }
        return message

    return build_node_sampler(dparsed["date_parsed"],  # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


# Paths and Neighbors -- Any Data Tyep

def find_unique_rels():
    """Fetch unique relationships that have a given node instance."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""How many unique relationships originate from {label_1} where {prop_1} is {val_1}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[r]->() RETURN COUNT(DISTINCT TYPE(r)) AS rels, TYPE(r)"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def connection_thru_two_rels():
    """How many nodes are connected to a given node instance via two relationships."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""How many nodes are connected to {label_1} for which {prop_1} is {val_1}, by exactly two different types of relationships?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[r]->(n) WITH n, COLLECT(DISTINCT TYPE(r)) AS Types WHERE SIZE(Types) = 2 RETURN COUNT(n)"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def rels_and_counts_and_nodes():
    """Get information on nodes connected to a certain node instance."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""List the nodes that are connected to {label_1} for which {prop_1} is {val_1}, with their relationship types and count these types!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[r]->(n) RETURN n, TYPE(r) AS Relations, COUNT(r) AS Counts"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def rels_and_counts():
    """Find relationships and their counts that are connected to a specified node instance."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""List the types of relationships and their counts connected to {label_1} for which {prop_1} is {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[r]->() RETURN TYPE(r) AS Relations, COUNT(r) AS Counts"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def find_node_neighbours():
    """Find all neighbors of a given node instance."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find all nodes directly connected to the {label_1} that has {prop_1} {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH path=(:{label_1} {{{prop_1}:'{val_1}'}})-->() RETURN path"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def find_neighbors_properties():
    """Find the neighbors of a given node (specified intrinsically) and list their properties."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the nodes connected to {label_1} where {prop_1} is {val_1} and list their properties!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[r]->(n) RETURN properties(n), r"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def find_node_neighbors_properties():
    """Find the neighbors of a given node (specified extrinsically) and list their properties."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Identify nodes that are connected to {label_1} where {prop_1} is {val_1} and list their properties, including those of {label_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (b:{label_1})-[r]->(n) WHERE b.{prop_1} = '{val_1}' RETURN properties(b) AS {label_1}_props, properties(n) AS props"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def find_properties_neighbors_relationship():
    """Find properties of specified neighbors of a given node instance."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""What are the properties of nodes connected to {label_1} for which {prop_1} is {val_1}, and what are their relationships to {label_1}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (c:{label_1})<-[r]-(n) WHERE c.{prop_1} = '{val_1}' RETURN properties(n) AS props, r"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def nodes_connected_to_two_nodes():
    """Find common neighbors of two nodes, only one specified."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Which nodes are connected to {label_1} where {prop_1} is {val_1}, and also to another node?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[r]->(n), (n)-[s]->(m) RETURN labels(n) AS Interim, labels(m) AS Target"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def longest_path_from_node():
    """Find the longest path originating from a given node, basic approach."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Identify the longest path originating from {label_1} for which {prop_1} is {val_1}, and list the properties of the nodes on the path!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH p=(a:{label_1}{{{prop_1}:'{val_1}'}})-[*]->(n) RETURN p, nodes(p) ORDER BY LENGTH(p) DESC LIMIT 1"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def node_properties_for_two_relationships():
    """Fetch node properties for a given path."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""What are the properties of nodes connected to {label_1} where {prop_1} is {val_1}, by two different types of relationships?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (e:{label_1}{{{prop_1}:'{val_1}'}})-[r1]->(n)-[r2]->(m) WHERE TYPE(r1) <> TYPE(r2) RETURN properties(n) AS props1, properties(m) AS props2"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def average_props():
    """Find the average count of properties of nodes along a path."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""What is the average number of properties per node connected to {label_1} for which {prop_1} is {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[r]->(n) RETURN AVG(SIZE(keys(n))) AS AvgProps"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def first_and_far_neighbors():
    """Proprieties of nodes for which there is a path to a specified node."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Enumerate the properties of nodes that are either directly or indirectly connected to {label_1} for which {prop_1} is {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[*]->(n) RETURN DISTINCT properties(n) AS Properties"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def nodes_connected_to_node():
    """Find the neighbors of a node (extrinsincally specified property)."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f""" List all nodes that are connected to {label_1} where {prop_1} contains {val_1}, along with the type of their relationship with {label_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"""MATCH (d:{label_1})-[r]->(n) WHERE d.{prop_1} CONTAINS '{val_1}' RETURN n, TYPE(r)"""
                           }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def find_far_unique_rels():
    """Find the distinct properties of nodes that are nhops away from a given node."""

    def prompter(*params, **kwargs):

        label_1= params[0]
        prop_1 = params[1]
        val_1 = params[2]
        nhops = kwargs.get('nhops', 2)


        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""List the distinct properties of nodes that are {nhops} hops away from {label_1} with {prop_1} equal to {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[*{nhops}]->(n) RETURN DISTINCT properties(n) AS props"
                   }
        return message

    sampler = []

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def find_far_neighbors_properties():
    """Find the properties of nodes that are 3 hops away from a given node instance."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        nhops = kwargs.get('nhops', 3)

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""List the properties of nodes that are {nhops} hops away from {label_1} with {prop_1} equal to {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f" MATCH (a:{label_1})-[*{nhops}]->(n) WHERE a.{prop_1} = '{val_1}' RETURN properties(n) AS props"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


def find_far_neighbors():
    """Retrieve the node labels of the nodes that are nhops away from a given node instance."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        nhops = kwargs.get('nhops', 3)

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""List nodes that are {nhops} hops away from {label_1} for which {prop_1}={val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[*{nhops}]->(n) RETURN labels(n) AS FarNodes"
                   }
        return message

    return build_node_sampler(dparsed["dtypes_parsed"],
                              prompter,
                              allow_repeats= ALLOW_REPEATS)


# One Node Label, Two Properties -----

# String Data Type ---

def match_with_where_not_value():
    """Retrieve a node property when another property does not take a certain value."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Retrieve distinct values of the {prop_2} from {label_1} where {prop_1} is not {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} <> '{val_1}' RETURN DISTINCT n.{prop_2} AS {prop_2}"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


def match_with_where_contains_substring():
    """Retrieve two properties of a node if one of the properties does contain a given substring."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {prop_1} and the {prop_2} for those {label_1} where {prop_1} contains the substring {val_1[:2]}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} CONTAINS '{val_1[2:]}' RETURN n.{prop_1} AS {prop_1}, n.{prop_2} AS {prop_2}"
        }
        return message

    return build_nodes_property_pairs_sampler(dparsed["string_parsed"],
                                              dparsed["dtypes_parsed"],
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)

def match_with_where_starts_with_substring():
    """Retrieve two properties of a node if one of the properties starts with a given substring."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {prop_1} and the {prop_2} for those {label_1} where {prop_1} starts with {val_1[0]}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} STARTS WITH '{val_1[0]}' RETURN n.{prop_1} AS {prop_1}, n.{prop_2} AS {prop_2}"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["string_parsed"],
                                              dparsed["dtypes_parsed"],
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


def match_with_where_not_is_value():
    """Return two properties of a node if one of the properties does not start with a specified string."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch unique values of {prop_1} and {prop_2} from {label_1} where {prop_1} does not start with {val_1[0]}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE NOT n.{prop_1} STARTS WITH '{val_1[0]}' RETURN DISTINCT n.{prop_1} AS {prop_1}, n.{prop_2} AS {prop_2}"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["string_parsed"],
                                              dparsed["dtypes_parsed"],
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


def match_properties_with_union():
    """Find node instances if one of two properties contains a certain substring."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Retrieve the {label_1} where {prop_1} or {prop_2} contains {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} CONTAINS '{val_1}' RETURN n AS node UNION ALL MATCH (m:{label_1}) WHERE m.{prop_2} CONTAINS '{val_1}' RETURN m AS node"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["string_parsed"],
                                              dparsed["string_parsed"],
                                       prompter,
                                       same_node = True,
                                       allow_repeats = ALLOW_REPEATS)


def where_one_node_two_props_notnull_or():
    """Find a specified property of a given label if another property fulfills a given condition or the specified property is not null."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch the distinct values of the {prop_2} from {label_1} where either {prop_1} is {val_1} or {prop_2} is not null!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} = '{val_1}' OR n.{prop_2} IS NOT NULL RETURN DISTINCT n.{prop_2} AS {prop_2}"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["string_parsed"],
                                              dparsed["string_parsed"],
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


# Temporal Data Type ---

def find_property_in_year():
    """Find a property of a given node if a temporal condition on a second property holds."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find all {prop_1} for {label_1} that have {prop_2} in 2020!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE date(n.{prop_2}).year = 2020 RETURN n.{prop_1}"
        }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["date_parsed"],  # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


def find_property_in_month():
    """Find how many nodes of have a first property and a temporal condition on a second property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find how many {label_1} with {prop_1} recorded have {prop_2} in June!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NOT NULL AND date(n.{prop_2}).month = 6 RETURN count(n)"
        }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["date_parsed"],  # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


def where_one_node_two_props_two_vals_or_notnull_date():
    """Find a temporal property for a specified node label when a second property takes a given value."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        prop_2 = params[3]
        val_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {prop_2} for those {label_1}s where {prop_1} is {val_1} and the year of the {prop_2} is {val_2[:4]}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} = '{val_1}' AND date(n.{prop_2}).year = {val_2[:4]} RETURN n.{prop_2} AS {prop_2}"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["date_parsed"], # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


def find_property_after_date():
    """Find a temporal property of a given node if a second temporal property satisfies a certain condition."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find all {prop_1} for {label_1} that have {prop_2} after January 1, 2020!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE date(n.{prop_2}) > date('2020-01-01') RETURN n.{prop_1}"
        }
        return message

    return build_nodes_property_pairs_sampler(dparsed["date_parsed"], # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                                              dparsed["date_parsed"], # dparsed["date_parsed"]+dparsed["date_time_parsed"] when available
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


# Numeric Data Type --- 

# Node count by property and relation
def aggregate_integers_by_string():
    """Find statistics of a numerical property for those nodes that satisfy a condition on a second property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""For each nonull {prop_1} of the {label_1}, how many times does it appear, and what are the minimum, maximum and average values of {prop_2} associated to it?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} IS NOT NULL WITH DISTINCT n WITH n.{prop_1} as {prop_1}, COUNT(n) AS count, min(n.{prop_2}) AS min, max(n.{prop_2}) AS max, avg(n.{prop_2}) AS avg RETURN {prop_1}, count, min, max, avg"
        }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["integer_parsed"], # dparsed["integer_parsed"]+dparsed["float_parsed"] when available
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


def match_with_where_not_null():
    """Return nodes where a property is not null, a second property takes specified values, order by the second property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        prop_2 = params[3]
        val_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Search for {prop_1} and {prop_2} from {label_1} where {prop_1} is not null and {prop_2} exceeds {val_2} and sort the results by {prop_2}, beginning with the largest!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1}  IS NOT NULL AND n.{prop_2} > {val_2} RETURN n.{prop_1} AS {prop_1}, n.{prop_2} AS {prop_2} ORDER BY {prop_2} DESC"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["string_parsed"],
                                              dparsed["integer_parsed"], # dparsed["integer_parsed"]+dparsed["float_parsed"] when available
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


def aggregate_numerical_by_integer():
    """Count the nodes where two properties satisfy two numerical conditions."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        prop_2 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {label_1} counts where {prop_1} is smaller than ten, and return the maximum, minimum and average values of the {prop_2}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} > 100 WITH DISTINCT n WITH n.{prop_1} as {prop_1}, COUNT(n) AS count, min(n.{prop_2}) AS min_{prop_2}, max(n.{prop_2}) AS max_{prop_2}, avg(n.{prop_2}) AS avg_{prop_2} RETURN {prop_1}, count, min_{prop_2}, max_{prop_2}, avg_{prop_2}"
        }
        return message

    return build_nodes_property_pairs_sampler(dparsed["integer_parsed"], # dparsed["integer_parsed"]+dparsed["float_parsed"] when available
                                              dparsed["integer_parsed"], # dparsed["integer_parsed"]+dparsed["float_parsed"] when available
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


def match_with_where_or_numerical_literal():
    """Find at most n nodes of specified label where a numerical property is greater or another is less than specified values."""
    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        prop_2 = params[3]
        val_2 = params[4]


        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_1, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find eight instances of {label_1} where either {prop_1} exceeds {val_1} or {prop_2} is less than {val_2}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE n.{prop_1} > {val_1} OR n.{prop_2} < {val_2} RETURN n LIMIT 8"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["integer_parsed"], # dparsed["integer_parsed"]+dparsed["float_parsed"] when available
                                              dparsed["integer_parsed"], # dparsed["integer_parsed"]+dparsed["float_parsed"] when available
                                              prompter,
                                              same_node=True,
                                              allow_repeats=ALLOW_REPEATS)


# Two Node Labels and Properties -----

# Relationships to Nodes ---

def find_nodes_connected_to_two_nodes():
    """Find the nodes connected to two given nodes."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        label_2 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, ], [label_2, ]], [], False, False, False)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find nodes that share a relationship with both {label_1} and {label_2}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"""MATCH (c:{label_1})<-[r1]-(n)-[r2]->(d:{label_2}) RETURN labels(n)"""
                           }
        return message

    return build_nodes_pairs(nodes,
                             prompter,
                             allow_repeats = ALLOW_REPEATS
                             )


def nodes_connected_to_two_nodes_both():
    """Find nodes on paths between two given nodes."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        label_2 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, ], [label_2, ]], [], False, False, False)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Identify nodes that are connected to both {label_1} and {label_2}, directly or indirectly!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (a:{label_1})-[*]-(n)-[*]-(b:{label_2}) RETURN labels(n)"
        }
        return message

    return build_nodes_pairs(nodes,
                             prompter,
                             allow_repeats = ALLOW_REPEATS
                             )


def find_common_rels():
    """Find nodes that share common relationships with two given nodes."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        label_2 = params[1]

        subschema = build_minimal_subschema(jschema, [[label_1, ], [label_2, ]], [], False, False, False)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Are there any nodes that share a common relationship type with both {label_1} and {label_2}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (a:{label_1})-[r]->(n), (d:{label_2})-[s]->(m) WHERE TYPE(r) = TYPE(s) RETURN labels(n), labels(m)"
                   }
        return message

    return build_nodes_pairs(nodes,
                             prompter,
                             allow_repeats = ALLOW_REPEATS
                             )


def rel_and_common_prop():
    """Identify nodes with common properties."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        label_2 = params[3]
        prop_2 = params[4]
        val_2 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Are there any nodes that are connected with {label_1} where {prop_1} is {val_1} and share a common property with {label_2}, for which {prop_2} equals {val_2}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"""MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[r]->(n), (d:{label_2}{{{prop_2}:'{val_2}'}}) WHERE ANY(key in keys(n) WHERE n[key] = d[key]) RETURN n"""}
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                       prompter,
                                       same_node = False,
                                       allow_repeats = ALLOW_REPEATS)


# Unions of Sets ---

def match_nodes_with_union_all():
    """Build a union of two sets (without filtering duplicates) extracted from two distinct node labels and their properties."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        label_2 = params[3]
        prop_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Return the {prop_1} for {label_1} combined with the {prop_2} for {label_2}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) RETURN n.{prop_1} AS Records UNION ALL MATCH (m:{label_2}) RETURN m.{prop_2} AS Records"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                       prompter,
                                       same_node = False,
                                       allow_repeats = ALLOW_REPEATS)


def match_nodes_with_union():
    """Build a union of two sets (with filtering duplicates) extracted from two distinct node labels and their properties."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        label_2 = params[3]
        prop_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Return the {prop_1} for {label_1} combined with the {prop_2} for {label_2}, filter the duplicates if any!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) RETURN n.{prop_1} AS Records UNION MATCH (m:{label_2}) RETURN m.{prop_2} AS Records"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                       prompter,
                                       same_node = False,
                                       allow_repeats = ALLOW_REPEATS)


# Retrieve Properties -----

def match_two_nodes_two_props():
    """Retrieve several samples of properties values that correspond to two node labels (same or distinct)."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        label_2 = params[3]
        prop_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch eight samples of the {prop_1} of the {label_1} and the {prop_2} for {label_2}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) MATCH (m:{label_2}) RETURN n.{prop_1}, m.{prop_2} LIMIT 8"
                   }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                       prompter,
                                       same_node = False,
                                       allow_repeats = ALLOW_REPEATS)


def where_not_simple_path_and_property():
    """Retrieve one property that is not in relationship to another node with a given property."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        label_2 = params[3]
        prop_2 = params[4]
        val_2 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Look for the {prop_1} of the {label_1} that is not related  to the {label_2} with the  {prop_2}  {val_2}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}), (:{label_2} {{{prop_2}: '{val_2}'}}) WHERE NOT (n) --> (:{label_2}) RETURN n.{prop_1}"
        }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                       prompter,
                                       same_node = False,
                                       allow_repeats = ALLOW_REPEATS)


# Paths ---

def path_existence():
    """Determine if there is a path connected two given nodes."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        label_2 = params[3]
        prop_2 = params[4]
        val_2 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Is there a path connecting {label_1} where {prop_1} is {val_1} and {label_2}, for which {prop_2} is {val_2}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"""MATCH (a:{label_1}{{{prop_1}:'{val_1}'}}), (b:{label_2}{{{prop_2}:'{val_2}'}}) RETURN EXISTS((a)-[*]-(b)) AS pathExists"""}
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                       prompter,
                                       same_node = False,
                                       allow_repeats = ALLOW_REPEATS)


def number_of_paths():
    """Find the number of paths with given end nodes."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        label_2 = params[3]
        prop_2 = params[4]
        val_2 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""How many paths are there between {label_1} where {prop_1} is {val_1} and {label_2}, for which {prop_2} equals {val_2}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"""MATCH p=(a:{label_1}{{{prop_1}:'{val_1}'}})-[*]->(d:{label_2}{{{prop_2}:'{val_2}'}}) RETURN count(p)"""}
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                       prompter,
                                       same_node = False,
                                       allow_repeats = ALLOW_REPEATS)


def end_of_the_path():
    """Find the end node of a given path."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        label_2 = params[3]
        prop_2 = params[4]
        val_2 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find nodes that are at the end of a path starting at {label_1} where {prop_1} is {val_1} and traversing through {label_2} with {prop_2} {val_2}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"""MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[*]->(d:{label_2}{{{prop_2}:'{val_2}'}})-[*]->(n) RETURN n
                    """ }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                       prompter,
                                       same_node = False,
                                       allow_repeats = ALLOW_REPEATS)


def shortest_path_between_two_nodes():
    """Find the shortest path between two nodes."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        label_2 = params[3]
        prop_2 = params[4]
        val_2 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2]], [], True, False, True)[:-29] # remove relationship comment
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the shortest path between {label_1} where {prop_1} is {val_1} and {label_2}, with {prop_2} equal {val_2}, including the nodes on the path!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"""MATCH p=shortestPath((a:{label_1}{{{prop_1}:'{val_1}'}})-[*]-(e:{label_2}{{{prop_2}:'{val_2}'}})) RETURN nodes(p)
                    """
                           }
        return message

    return build_nodes_property_pairs_sampler(dparsed["dtypes_parsed"],
                                              dparsed["dtypes_parsed"],
                                       prompter,
                                       same_node = False,
                                       allow_repeats = ALLOW_REPEATS)


# Relationships -----

# Nodes and Relationships ---

def find_not_connected_nodes():
    """Identify nodes that do not have certain relationships."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        rel_1 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, ]], [[rel_1, ]], False, False, False)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch five {label_1} that are not linked through {rel_1} relationships!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (p:{label_1}) WHERE NOT EXISTS ((p)-[:{rel_1}]->()) RETURN p LIMIT 5"
        }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_connected_nodes():
    """Find nodes that are connected via certain relationships."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        rel_1 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, ]], [[rel_1, ]], False, False, False)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find four {label_1} that have {rel_1} links!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (p:{label_1}) WHERE EXISTS ((p)-[:{rel_1}]->()) RETURN p LIMIT 4"
        }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_relation_count():
    """Count the number of specified relationships a node has."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        rel_1 = params[3]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1]], [[rel_1, ]], True, False, False)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch ten {label_1} and return the {prop_1} and the number of nodes connected to them via {rel_1} given in descending order of the node counts.""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WITH n.{prop_1} AS {prop_1}, size([(n)-[:{rel_1}]->() | 1]) AS count ORDER BY count DESC LIMIT 10 RETURN article_id, count"
                   }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


# Two Labels, One Property ---

def nodes_connected_to_first_node_and_not_connected_to_second_node():
    """Determine which nodes are connected to node A but not connected to node B via a given relationship."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1,], [label_2, ]], [[rel_1, ]], False, False, False)
        message = {"Prompt": f"{system_message}",
                   "Question": f""" Which nodes are connected to {label_1}, but not to {label_2} via {rel_1}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"""MATCH (c:{label_1})-[r]-(n) WHERE NOT (n)-[:{rel_1}]-(:{label_2}) RETURN labels(n)"""
                           }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_property_with_count_limit():
    """Retrieve property values for several nodes A and the number of relationship counts to nodes B."""
    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Search for the {prop_1} values from 20 {label_1} that are linked to {label_2} via {rel_1} and return {prop_1} along with the respective {label_2} counts!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[:{rel_1}]->(m:{label_2}) WITH DISTINCT n, m RETURN n.{prop_1} AS {prop_1}, count(m) AS count LIMIT 20"
        }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_property_by_condition_on_node():
    """Retrieve property values for nodes A that have more than five relationships to nodes B."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {prop_1} of {label_1} that each have more than five {rel_1} relationships with {label_2}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[r:{rel_1}]->(m:{label_2}) WITH DISTINCT n, m, r WITH n.{prop_1} AS {prop_1}, count(r) AS count WHERE count > 5 RETURN {prop_1}"
        }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def where_and_exists_simple_path():
    """Fetch a property of nodes connected to a given node via a specified relationship."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch {prop_1} of the {label_1} that are connected to {label_2} via {rel_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE EXISTS {{ MATCH (n)-[:{rel_1}]->(:{label_2}) }} RETURN n.{prop_1} AS {prop_1}"}
        return message

    return build_relationships_samples(drels["string_string_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_relation_ordered_count_desc():
    """Retrieve, in descending order, the count of nodes linked to a given node."""
    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""For each {label_1} find its {prop_1} and the count of {label_2} linked via {rel_1}, and retrieve seven results in desc order of the counts!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[:{rel_1}]->(m:{label_2}) WITH DISTINCT n, m RETURN n.{prop_1} AS {prop_1}, count(m) AS count ORDER BY count DESC LIMIT 7"
        }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_relation_ordered_count():
    """Retrieve, in ascending order, the counts of nodes linked to a given node."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""For each {label_1}, find the number of {label_2} linked via {rel_1} and retrieve the {prop_1} of the {label_1} and the {label_2} counts in ascending order!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[:{rel_1}]->(m:{label_2}) WITH DISTINCT n, m RETURN n.{prop_1} AS {prop_1}, count(m) AS {label_2.lower()}_count ORDER BY {label_2.lower()}_count"
        }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_relation_ordered_count_filter():
    """Retrieve the counts, larger than a given value, of nodes linked to a given node."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""For each {label_1} and its {prop_1}, count the {label_2} connected through {rel_1} and fetch the {prop_1} and the counts that are greater than 5, starting with the largest {prop_1} and count!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[:{rel_1}]->(m:{label_2}) WITH DISTINCT n, m WITH n.{prop_1} AS {prop_1}, count(m) AS count WHERE count > 4 RETURN {prop_1}, count ORDER BY {prop_1} DESC, count DESC"
        }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_common_prop():
    """Find related nodes with common properties."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Which nodes have a common property with {label_1} where {prop_1} is {val_1} and are {rel_1} linked to a {label_2}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (a:{label_1} {{{prop_1}:'{val_1}'}})-[r:{rel_1}]->(b:{label_2}) WHERE ANY(key IN keys(a) WHERE a[key] = b[key]) RETURN b"
                   }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_end_nodes_path():
    """Find nodes that are at the end of a path with specified starting node."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Which nodes are at the end of a path starting from {label_1}, with {prop_1} equal to  {val_1}, passing through {label_2} via {rel_1}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"""MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[:{rel_1}]->(c:{label_2})-[r]->(n) RETURN n"""
                           }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


# Extract properties of end nodes of a path
def find_end_node_properties():
    """Find properties of nodes connected to specified nodes."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        label_2 = params[4]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""What are the properties of {label_2} that is {rel_1} connected to {label_1} that has {prop_1} equal to {val_1}?""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[:{rel_1}]->(m:{label_2}) WHERE n.{prop_1} = {val_1} RETURN properties(m) AS props"
                   }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


# Two Labels, Two Properties ---

def find_node_relation_ordered_count_collect():
    """Find properties of nodes that are related under given conditions."""

    def prompter(*params, **kwargs):

        label_1 = params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        label_2 = params[4]
        prop_2 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2 ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch the {prop_1} of the {label_1} that are linked via {rel_1} to more than three {label_2}, and list {label_2} {prop_2} and {label_2} counts, ordering by {label_2} count and limiting to the top six results!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[:{rel_1}]->(m:{label_2}) WITH DISTINCT n, m WITH n.{prop_1} AS {prop_1}, count(m) AS count, COLLECT(m.{prop_2}) as {prop_2} WHERE count > 3 RETURN {prop_1}, count, {prop_2} ORDER BY count LIMIT 6"
        }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


# Node count by property and relation
def find_node_aggregation_date_rels():
    """Evaluate the average values of a property for all nodes of the same label that are connected to a specified node."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        label_2 = params[4]
        prop_2 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2 ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Calculate the average {prop_2} for {label_2} that are linked to {label_1} via {rel_1} and have {prop_1} date before December 31, 2020!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[:{rel_1}]->(m:{label_2}) WHERE m.{prop_1} < date('2020-12-31') RETURN avg(m.{prop_2}) AS avg_{prop_2}"
        }
        return message

    return build_relationships_samples(drels["all_rels"],  # best with drels["date_integer_rels"] or drels["date_float_rels"] if available
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def where_and_simple_path():
    """Find a property of a node connected via a given relationship to a node for which a certain property takes a specified value."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        label_2 = params[4]
        prop_2 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2 ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": "Convert the following question into a Cypher query using the provided graph schema!",
                   "Question": f"""Retrieve the {prop_2} for {label_2} that is linked through a {rel_1} relationship with the {label_1} where {prop_1} is {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[{rel_1[:2].lower()}:{rel_1}]->(m) WHERE n.{prop_1}='{val_1}' RETURN m.{prop_2}"
                   }
        return message

    return build_relationships_samples(drels["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def relation_with_and_where():
    """Retrieve related node properties that satisfy given conditions."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        label_2 = params[4]
        prop_2 = params[5]
        val_2 = params[6]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2, prop_2 ]], [[rel_1, ]], True, False, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find {label_2} that has a {prop_2} which begins with {label_2[0].lower()}, and is linked to {label_1} via {rel_1} relationship, where {label_1} has {prop_1} {val_1}!""",
                   "Schema": f"Graph schema: {subschema}",
                   "Cypher": f"MATCH (n:{label_1} {{{prop_1}: '{val_1}'}}) -[:{rel_1}]- (m:{label_2}) WHERE m.{prop_2} STARTS WITH '{label_2[0].lower()}' RETURN m"
        }
        return message

    return build_relationships_samples(drels["string_string_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


# Relationships with Properties -----

# Nodes and Relationships (With Properties) ---

# Pattern check
def find_not_connected_nodes_relprops():
    """Identify nodes that do not have certain relationships."""

    def prompter(*params, **kwardgs):

        label_1= params[0]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, ]], [[rel_1,rprop_1]],False, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch five {label_1} that are not linked through {rel_1} relationships where {rprop_1} is {rval_1}!""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (p:{label_1}) WHERE NOT EXISTS {{(p)-[r:{rel_1}]->() WHERE r.{rprop_1}='{rval_1}' }} RETURN p LIMIT 5"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_connected_nodes_relprops():
    """Find nodes that are connected via certain relationships."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1, ]], [[rel_1,rprop_1 ]],False, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find four {label_1} that have {rel_1} links so that {rprop_1} are {rval_1}!""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (p:{label_1}) WHERE EXISTS {{(p)-[r:{rel_1}]->() WHERE r.{rprop_1}='{rval_1}'}}  RETURN p LIMIT 4"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_relation_count_relprops():
    """Count the number of specified relationships a node has."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]

        subschema = build_minimal_subschema(jschema, [[label_1,prop_1 ]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch all the {label_1} and return the {prop_1} and the number of nodes connected to them via {rel_1} with {rprop_1} = {rval_1}.""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (n:{label_1})-[r:{rel_1}]->() WHERE r.{rprop_1} = '{rval_1}' WITH (n), COUNT(*) AS numberOfDirectConnections RETURN n.{prop_1} AS {prop_1}, numberOfDirectConnections"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


# Two Labels, One Property, Relationship (with property) ---

def find_node_property_with_count_limit_relprops():
    """Retrieve property values for several nodes A and the number of relationship counts to nodes B."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question":  f"""Search for the {prop_1} values from 20 {label_1} that are linked to {label_2} via {rel_1} with {rprop_1} = {rval_1}, and return {prop_1} along with the respective {label_2} counts!""",
                   "Schema":f"{subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[r:{rel_1}]->(m:{label_2}) WHERE r.{rprop_1}='{rval_1}' WITH DISTINCT n, m RETURN n.{prop_1} AS {prop_1}, count(m) AS count LIMIT 20"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def where_and_exists_simple_path_relprops():
    """Fetch a property of nodes connected to a given node via a relationship with specified properties."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Fetch {prop_1} of the {label_1} that are connected to {label_2} via {rel_1} where {rprop_1} are at most {rval_1}!""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (n:{label_1}) WHERE EXISTS {{ MATCH (n)-[r:{rel_1}]->(:{label_2}) WHERE r.{rprop_1} < '{rval_1}'}} RETURN n.{prop_1} AS {prop_1}"}

        return message

    return build_relationships_props_samples(drelsprops["all_rels"], # use drelsprops["string_integer_string_rels"] if available
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_relation_ordered_count_desc_relprops():
    """Retrieve, in descending order, the count of nodes linked to a given node, via a specified relationship."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""For each {label_1} find its {prop_1} and the count of {label_2} linked via {rel_1} where {rprop_1} is not '{rval_1}', and retrieve seven results in desc order of the counts!""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[r:{rel_1}]->(m:{label_2}) WHERE r.{rprop_1} <> '{rval_1}' WITH DISTINCT n, m RETURN n.{prop_1} AS {prop_1}, count(m) AS count ORDER BY count DESC LIMIT 7"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_relation_ordered_count_relprops():
    """Retrieve, a property and the counts, in ascending order, of nodes linked to a given node."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""For each {label_1}, find the number of {label_2} linked via {rel_1} where {rprop_1} is {rval_1} and retrieve the {prop_1} of the {label_1} and the {label_2} counts in ascending order!""",
                   "Schema": f"{subschema}",
                    "Cypher": f"MATCH (n:{label_1}) -[r:{rel_1}]->(m:{label_2}) WHERE r.{rprop_1} = '{rval_1}' WITH DISTINCT n, m RETURN n.{prop_1} AS {prop_1}, count(m) AS count ORDER BY count"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_common_prop_relprops():
    """Find related nodes with common properties related via a specified relationship."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Which nodes have a common property with {label_1} where {prop_1} is {val_1} and are {rel_1} linked to {label_2}, where {rprop_1} is {rval_1}?""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (a:{label_1}{{{prop_1}:'{val_1}'}})-[r:{rel_1} {{{rprop_1} :'{rval_1}'}}]->(b:{label_2}) WHERE ANY(key IN keys(a) WHERE a[key] = b[key]) RETURN b"
                   }

        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_end_nodes_path_relprops():
    """Find nodes that are at the end of a path with specified starting node and interim relationship."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Which nodes are at the end of a path starting from {label_1}, where {prop_1} is {val_1}, through {label_2} via {rel_1} with {rprop_1} {rval_1}?""",
                   "Schema": f"{subschema}",
                   "Cypher": f"""MATCH (a:{label_1} {{{prop_1}:'{val_1}'}})-[:{rel_1} {{{rprop_1}: '{rval_1}'}}]->(c:{label_2})-[r]->(n) RETURN n"""
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_end_node_properties_relprops():
    """Find properties of nodes connected to specified nodes, via specified relationship."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""What are the properties of {label_2} that is {rel_1}, with {rprop_1} equal to {rval_1}, connected to {label_1} that has {prop_1} equal to {val_1}?""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[:{rel_1}{{{rprop_1}: '{rval_1}'}}]->(m:{label_2}) WHERE n.{prop_1} = '{val_1}' RETURN properties(m) AS props"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


# Two Labels, Two Properties, Relationship (with property) ---

def find_node_relation_node_count_relprops():
    """Retrieve properties and counts of nodes connected via a specified relationship."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find {prop_1} of the {label_1} and return it along with the count of {label_2} that are linked via {rel_1} where {rprop_1} is {rval_1}!""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[r:{rel_1}]->(m:{label_2}) WHERE r.{rprop_1} = '{rval_1}' RETURN n.{prop_1} AS {prop_1}, count(m) AS count"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def relation_with_and_where_relprops():
    """Find node properties that are connected via a relationship with non-null property."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        rprop_1 = params[4]
        label_2 = params[6]
        prop_2 = params[7]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,prop_2]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Find the {label_2} with a {prop_2} starting with {label_2[0]}, and linked with an {label_1} through {rel_1} relationship. The {label_1} must have {prop_1}: {val_1} and be {rel_1} with {rprop_1} recorded!""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (n:{label_1} {{{prop_1}: '{val_1}'}}) -[r:{rel_1}]- (m:{label_2}) WHERE m.{prop_2} STARTS WITH '{label_2[0]}' AND r.{rprop_1} IS NOT NULL RETURN n.{prop_2}"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def find_node_aggregation_date_rels_relprops():
    """Find property average of a node in a specified relationship with another given node."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]
        prop_2 = params[7]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,prop_2]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question": f"""Calculate the average {prop_2} for {label_2} that is linked to {label_1} via {rel_1} where {rprop_1} is {rval_1} and has {prop_1} date before December 31, 2020!""",
                   "Schema": f"{subschema}",
                   "Cypher": f"MATCH (n:{label_1}) -[:{rel_1}{{{rprop_1}: '{rval_1}'}}]->(m:{label_2}) WHERE m.{prop_1} < date('2020-12-31') RETURN avg(m.{prop_2}) AS avg_{prop_2}"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"],  # best with drelsprops["date_string_integer_rels"] when available
                            prompter,
                            allow_repeats=ALLOW_REPEATS)


def where_and_simple_path_relprops():
    """Retrieve properties of specific nodes that have relationships with given properties."""

    def prompter(*params, **kwardgs):

        label_1 = params[0]
        prop_1 = params[1]
        val_1 = params[2]
        rel_1 = params[3]
        rprop_1 = params[4]
        rval_1 = params[5]
        label_2 = params[6]
        prop_2 = params[7]

        subschema = build_minimal_subschema(jschema, [[label_1, prop_1], [label_2,prop_2]], [[rel_1,rprop_1 ]],True, True, True)
        message = {"Prompt": f"{system_message}",
                   "Question":  f"""Search for the {prop_2} in {label_2} that is linked through a {rel_1} relationship with {label_1} where {prop_1} is {val_1} and {rel_1} has {rprop_1} on {rval_1}!""",
                   "Schema": f"{subschema}",
                   "Cypher":  f"MATCH (n:{label_1}) -[{rel_1[:2].lower()}:{rel_1} {{{rprop_1} : '{rval_1}'}}]->(m) WHERE n.{prop_1}='{val_1}' RETURN m.{prop_2}"
        }
        return message

    return build_relationships_props_samples(drelsprops["all_rels"], #"string_date_string_rels"],
                            prompter,
                            allow_repeats=ALLOW_REPEATS)



# END of builders 
# ---------------------------------------------


def generate_samples():
    """
    Actually call the individual samplers to generate the samples
    """
    # List of samplers, as callables, to use to generate the samples.
    # By default we set it to all samplers defined, but this list can be modified as needed.
    
    # create a list of the actual functions defined in this module corresponding to the name
    # provided by the string, and strip off the newline characters from the string 
    samplers = [globals()[x.strip()] for x in all_samplers]
    
    # List to collect the samples
    trainer=[]

    for s in samplers:
        sampler = s()
        trainer += collect_samples(sampler, M)

    sampler = count_nodes_of_given_label()
    trainer += collect_samples(sampler, M)

    sampler = paths_with_node_endpoint()
    trainer += collect_samples(sampler, M)

    sampler = match_one_node_one_prop()
    trainer += collect_samples(sampler, M)

    sampler = where_one_node_one_prop_notnull_numeral()
    trainer += collect_samples(sampler, M)

    sampler = where_one_node_one_prop_notnull_literal()
    trainer += collect_samples(sampler, M)

    sampler = where_one_node_one_prop_null_numeral()
    trainer += collect_samples(sampler, M)

    sampler = find_node_property_count()
    trainer += collect_samples(sampler, M)

    sampler = find_node_property_count()
    trainer += collect_samples(sampler, M)

    sampler = find_node_by_property()
    trainer += collect_samples(sampler, M)

    sampler = match_skip_limit_return_property()
    trainer += collect_samples(sampler, M)


    # Display the number of samples created and save the data to a file
    print(f"There are {len(trainer)} samples in the fine-tuning dataset.")

    # write samples to file
    print(f"\nWriting samples to path: {data_path+trainer_with_repeats_file}")
    write_json(trainer, data_path+trainer_with_repeats_file)


def create_min_samples_file(read_path, write_path):
    """
    Reads the training with repeats file and creates a minimal samples JSON file.
    """
    
    with open(read_path, 'r') as f: 
        data = json.load(f)
    result = []
    for d in data:
        result.append({"question": d['Question'], "query": d["Cypher"]})
    with open(write_path, 'w') as f:
        json.dump(result, f)
    print(f"A total of {len(result)} samples written to min file at path {write_path}")
    

def main():
    # gather_from_neo()
    get_nodes_props_instances_from_files()
    generate_samples()
    create_min_samples_file(data_path+trainer_with_repeats_file, data_path+"min_samples.json")


if __name__ == '__main__':
    main()