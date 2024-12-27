Hybrid Logic Graph (HyLo Graph)
===============================
Hybrid language-logic framework integrated via a directed graph execution engine.

See the [hylograph](hylograph/README.md) directory for an overview of the package and installation instructions.

Works with local or remote instances of models, such as Ollama.

To test with Ollama locally, follow the [documentation](https://hub.docker.com/r/ollama/ollama) to run with docker, e.g., 

```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

