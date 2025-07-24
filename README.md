# PSC-CMU-PITT-HACKATHON GraphRAG Project

This repository demonstrates a simple Graph Retrieval Augmented Generation (RAG) workflow. It contains scripts to build a small knowledge graph from text, query it using a language model served through [vLLM](https://github.com/vllm-project/vllm), and explore the results in a Jupyter notebook.

## Requirements

- **Python 3.8+**
- **CUDA-enabled GPU** (required by vLLM)
- All Python dependencies listed in `requirements.txt`

Install the Python packages with:

```bash
pip install -r requirements.txt
```

## Starting the vLLM Server

The `run_vllm.sh` script launches a vLLM API server using the `Qwen/Qwen2.5-7B-Instruct-AWQ` model. Make sure your GPU drivers are configured properly before starting the server. You can optionally enable FlashInfer by setting `VLLM_ATTENTION_BACKEND=FLASHINFER` before running the script.

```bash
bash run_vllm.sh
```

By default the server listens on port `8000`.

## Running the Notebook

The `docGraphRAG.ipynb` notebook walks through constructing a graph from text and interacting with the vLLM server. Start a Jupyter environment and open the notebook:

```bash
jupyter notebook docGraphRAG.ipynb
```

Follow the cells in order to load data, build the graph and generate answers.

