# Hybrid Logic Graph (HyLo Graph)

Hybrid language-logic framework integrated via a directed graph execution engine.


## Installation

This project uses Python Poetry. Install the package dependencies in a virtual environment
by executing:

```
poetry install --no-root
```

Then, activate the environment from the command line using:

```
poetry shell
```

See the [hylograph](hylograph/README.md) directory for an overview of the package and usage instructions.


This package can work with either local or remote instances of models, such as Ollama.

To test with Ollama locally, follow the [documentation](https://hub.docker.com/r/ollama/ollama) to run with docker, e.g., 

```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
Be sure to download the required models, e.g., 

```
docker exec -it ollama bash

ollama pull mxbai-embed-large
ollama pull llama3.1:8b
```

Also, make sure to check that the app's configuration is set to use a localhost URL 
for the models. For example, for the demo app, check the `model_base_url` property 
defined within the `apps/movie_demo.py` file. 