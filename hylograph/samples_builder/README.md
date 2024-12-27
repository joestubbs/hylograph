# Samples Builder
Automatically generate samples based on database
schema. This work is based on the Neo4JLabs [text2cypher](https://github.com/neo4j-labs/text2cypher/tree/main) code
repository.

## Usage

The main python program is [cypher_samples.py](cypher_samples.py)
in the current directory. At a high level, this program works in a 
series of steps:

Step 1: inspect a Neo4J database and write high-level features to a set
of JSON files. This is the only step that needs a connection to the 
Neo4J database. 

Step 2: From the JSON files in step 1, gathers initial data structures from the files.

Step 3: Iterate through a suite of configured "samplers" to generate
a set of samples.

Step 4: Write the generated samples to output files. A couple of 
different formats can be written to support different usages:
fine-tuning or benchmarking. 

Currently, this file should be modified with configuration for your
Neo4J database. See the `ToDo` comments in the file. 

Once the configuration has been updated in the file, simply 
execute it with python in a poetry environment created using the 
[pyproject.toml](../../pyproject.toml) at the project root.

```
python cypher_samples.py
```

Depending on options chosen, files will be written to the `data_path`
directory configured in the file. 