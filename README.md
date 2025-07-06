# GraphRAG: Graph-Based Retrieval-Augmented Generation for Biomedical Knowledge

GraphRAG combines a biomedical knowledge graph with retrieval-augmented generation to answer complex scientific questions. The project demonstrates a lightweight implementation using Python, Neo4j and the OpenAI API.

## Project Goals
- Leverage GPU acceleration for graph-based retrieval and LLM inference.
- Provide reproducible examples of graph construction and querying for biomedical datasets.
- Offer a foundation for researchers to extend and optimize large-scale knowledge graphs.

## Getting Started
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd PSC-CMU-Pitt-Hackathon
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the demo notebook**
   ```bash
   jupyter notebook notebooks/GraphRAG_demo.ipynb
   ```

The notebook now exports the Neo4j graph to a GML file and uses the
`IterativeKnowledgeGraphAgent` implemented in `graphreader.ipynb` (and
available as `iterative_graphreader.py`) to explore the graph.

## Scripts
- `graphrag_data_pipeline.py` – retrieves documents from PubMed, extracts entities with spaCy and loads them as nodes and relationships into Neo4j.
- `graphreader_agent.py` – a minimal multi-step reasoning agent that uses an LLM to formulate a plan, select nodes and produce an answer.
- `iterative_graphreader.py` – module generated from `graphreader.ipynb` providing the `IterativeKnowledgeGraphAgent`.
- `create_dynamic_notebook.py` – utility for generating a custom data collection notebook.

## Requirements
- Python 3.8+
- Neo4j running locally
- OpenAI API key (for LLM calls)

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
