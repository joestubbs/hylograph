"""
Module containing functions for evaluating graphs on datasets.

Based on the Neo4j text2cypher repository;

cf., https://github.com/neo4j-labs/text2cypher/blob/main/evaluations/evaluating_cypher_jaccard.ipynb

"""

import textdistance
import statistics
from typing import Set, Any, Union, Dict, List, Tuple, Hashable


def get_jw_distance(string1: str, string2: str) -> float:
    """
    Calculate the Jaro-Winkler distance between two strings.

    The Jaro-Winkler distance is a measure of similarity between two strings.
    The score is normalized such that 0 equates to no similarity and
    1 is an exact match.
    """
    # Call the jaro_winkler function from the textdistance library.
    return textdistance.jaro_winkler(string1, string2)


def rowsim(setL: Set, setR: Set) -> float:
    """
    Calculate the similarity between two sets using Jaccard index formula.
    """
    return len(setL.intersection(setR)) / len(setL.union(setR))


def floatify(v: Any) -> Any:
    """
    Attempts to convert a value to a float if it is a string and represents a
    number, or recursively apply the conversion to elements within a list or dict.
    """
    if isinstance(v, str):
        return v
    try:
        f = float(v)
        return f
    except:
        pass
    if isinstance(v, list):
        return [floatify(x) for x in v]
    if isinstance(v, dict):
        return {k: floatify(u) for k, u in v.items()}
    return v


def make_hashable(v: Any) -> Hashable:
    """
    Convert a value to a hashable type (needed for set operations).
    """
    float_v = floatify(v)
    if not isinstance(float_v, Hashable):
        return str(float_v)
    else:
        return float_v


def make_alignment(dictL: List[Dict], dictR: List[Dict]) -> Tuple[List[Set], List[Set]]:
    """
    Align rows from two lists of dictionaries based on their similarity.
    """
    swap = len(dictL) > len(dictR)

    # Forming set views from the list of dictionaries.
    setViewsL = [{make_hashable(v) for k, v in row.items()} for row in dictL]
    setViewsR = [{make_hashable(v) for k, v in row.items()} for row in dictR]
    if swap:
        setViewsL, setViewsR = setViewsR, setViewsL

    for i in range(len(setViewsL)):
        max_sim = -1
        max_j = -1
        for j in range(i, len(setViewsR)):
            sim = rowsim(setViewsL[i], setViewsR[j])
            if sim > max_sim:
                max_j = j
                max_sim = sim
        tmp = setViewsR[i]
        setViewsR[i] = setViewsR[max_j]
        setViewsR[max_j] = tmp
    if swap:
        setViewsL, setViewsR = setViewsR, setViewsL
    return setViewsL, setViewsR


def df_sim(dictL: List[Dict], dictR: List[Dict], list_view: bool) -> float:
    """
    Calculate the data frame similarity based on either the original row order or an alignment.
    """
    if list_view:
        # Original row order for lists of dictionaries
        view_L = [row.values() for row in dictL]
        view_R = [row.values() for row in dictR]
    else:
        view_L, view_R = make_alignment(dictL, dictR)

    totalSetL = set()
    for i, s in enumerate(view_L):
        for elem in s:
            totalSetL.add((i, make_hashable(elem)))
    totalSetR = set()
    for i, s in enumerate(view_R):
        for elem in s:
            totalSetR.add((i, make_hashable(elem)))
    intersection = totalSetL.intersection(totalSetR)
    union = totalSetL.union(totalSetR)

    if len(union) == 0 and len(intersection) == 0:
        return 1.0
    elif len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def df_sim_pair(pair_L, pair_R):
    """
    Compute the Jaccard similarity of two data frames (lists of dictionaries),
    taking into account the order of rows if indicated by the involved Cypher queries.
    """
    cypher_L, dict_L = pair_L
    cypher_R, dict_R = pair_R

    return df_sim(dict_L, dict_R, "order by" in f"{cypher_L} {cypher_R}".lower())


def evaluation(benchmark, invoke_model, graph, model_desc, out_file="output.csv"):
    """
    Primary entrypoint for the module; evaluates a model against a benchmark dataset. 
    
    Input:
    
    `benchmark`: A python list of dictionaries with keys `question` and `query`.
      * `question`: A question, in natural language, to send to the model.
      * `query`: The correct database query associated with the question.

    `graph`: A `Neo4jGraph` object, used with `query` to query the database to obtain the actual
        answer to the question.
    
    `invoke_model`: A callable that can be used to invoke the model against a natural language 
        question and return a cypher query. 
    
    `model_desc`: A description of the model, used for the report.
    
    Output:

    `results`: A dictionary of lists resulting from the evaluation. 
    
    """
    print("\n\n********** Starting Benchmark **********\n")
    # Create empty lists to store the results of the new columns
    generated_cyphers = []
    true_data = []
    eval_data = []
    jaro_winklers = []
    pass_1s = []
    pass_3s = []
    jaccards = []
    total_items = len(benchmark)

    # write header
    with open(out_file, 'w') as f:
        f.write(f"model_app_desc,natural_lang_question,model_generated_cyphers,model_generated_answer,true_answer,jaro_winkler,pass_1,pass_3,jaccard\n")

    # simpler report file that doesn't include the query
    out_file_simp = out_file.strip(".csv") + "_SIMPLE.csv"
    with open(out_file_simp, 'w') as f:
        f.write(f"model_app_desc,natural_lang_question,model_generated_answer,true_answer,jaro_winkler,pass_1,pass_3,jaccard\n")


    for idx, item in enumerate(benchmark):
        print(f"\n* * * Beginning benchmark for item {idx+1} out of {total_items}")
        # Fetch data based on the test Cypher statement
        true_data = graph.query(item["query"])

        # Generate 3 Cypher statement from model and fetch data
        example_generated_cyphers = []
        example_eval_data = []
        for _ in range(3):
            model_cypher = invoke_model(item["question"])
            example_generated_cyphers.append(model_cypher)
             # Fetch data based on the generated Cypher statement
            try:
                example_eval_data.append(graph.query(model_cypher))
            except ValueError:  # Handle syntax error
                example_eval_data.append([{"id": "Cypher syntax error"}])
    
        # These metrics require only the first cypher/response
        jaro_winkler = get_jw_distance(item["query"], example_generated_cyphers[0])
        pass_1 = (
            1
            if df_sim_pair(
                (item["query"], true_data),
                (example_generated_cyphers[0], example_eval_data[0]),
            )
            == 1
            else 0
        )
        jaccard = df_sim_pair(
            (item["query"], true_data),
            (example_generated_cyphers[0], example_eval_data[0]),
        )
        # Pass@3 check all 3 responses
        pass_3 = 1 if any(
            df_sim_pair((item["query"], true_data), (gen_cypher, eval_data)) == 1
            for gen_cypher, eval_data in zip(example_generated_cyphers, example_eval_data)
        ) else 0

        # Append the results to their respective lists
        generated_cyphers.append(example_generated_cyphers)
        true_data.append(true_data)
        eval_data.append(example_eval_data)
        jaro_winklers.append(jaro_winkler)
        pass_1s.append(pass_1)
        pass_3s.append(pass_3)
        jaccards.append(jaccard)
        
        # write result to report file
        print(f"\n***** Writing to {out_file} for item {idx+1} of {total_items}.")
        with open(out_file, 'a') as f:
            f.write(f"{model_desc},{item['question']},{example_generated_cyphers},{example_eval_data},{true_data},{jaro_winkler},{pass_1},{pass_3},{jaccard}\n")
        
        truncated_example_eval_data = str(example_eval_data)[0:20]
        with open(out_file_simp, 'a') as f:
            f.write(f"{model_desc},{item['question']},{truncated_example_eval_data},{true_data},{jaro_winkler},{pass_1},{pass_3},{jaccard}\n")

    # Compute and write report of means of metrics
    compute_and_write_means(jaro_winklers, pass_1s, pass_3s, jaccards, out_file)

    # Create and return the result object:
    result = {
        "generated_cypher": generated_cyphers,
        "true_data": true_data,
        "eval_data": eval_data,
        "jaro_winkler": jaro_winkler,
        "pass_1": pass_1,
        "pass_3": pass_3,
        "jaccard": jaccards,
    }

    return result


def compute_and_write_means(jaro_winklers, pass_1s, pass_3s, jaccards, outfile):

    # Calculate averages and write that as a separate report:
    avg = {
        "jaro_winkler": statistics.mean(jaro_winklers),
        "pass_1": statistics.mean(pass_1s),
        "pass_3": statistics.mean(pass_3s),
        "jaccard": statistics.mean(jaccards)
    }
    averages_file = outfile.strip(".csv") + "_AVGs.csv"
    print(f"Writing averages to {averages_file}...")
    with open(averages_file, 'w') as f:
        f.write(f"jaro_winkler,pass_1,pass_3,jaccard\n")
        f.write(f"{avg['jaro_winkler']},{avg['pass_1']},{avg['pass_3']},{avg['jaccard']}")

