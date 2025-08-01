# PSC-CMU-PITT-HACKATHON GraphRAG Project

This repository demonstrates a Graph Retrieval Augmented Generation (RAG) workflow that combines knowledge graph construction with iterative question answering. The system fetches scientific literature from multiple sources, builds a knowledge graph, and uses an intelligent agent to explore and answer complex questions.

## Features

- **Multi-source Data Retrieval**: Fetches abstracts from PubMed, Europe PMC, and preprint servers
- **Automated Knowledge Graph Construction**: Uses LLM-powered entity and relation extraction
- **Iterative Question Answering**: Smart agent that explores the graph step-by-step to answer complex queries
- **Multiple Variable Definitions**: Support for default and ontological entity definitions
- **vLLM Integration**: Inference using quantized Qwen2.5-7B model

## Requirements

- **Python 3.8+**
- **CUDA-enabled GPU** (required by vLLM)
- **16GB+ GPU memory** recommended for optimal performance
- All Python dependencies listed in `requirements.txt`

Install the Python packages with:

```bash
pip install -r requirements.txt
```

## Setup

**Important**: Before using the data retrieval functionality, you must configure your email address for PubMed API access:

1. Open `data_retriever.py`
2. Replace `"your_email@example.com"` with your actual email address on line 6:
   ```python
   Entrez.email = "your_email@example.com"  # Replace with your email
   ```

This is required by NCBI's Entrez API for responsible usage and is necessary for fetching PubMed abstracts.

## Project Structure

- **`data_retriever.py`**: Fetches scientific literature from PubMed, Europe PMC, and preprint servers
- **`kg_builder.py`**: Constructs knowledge graphs from text using LLM-powered extraction
- **`graphRAG_agent.py`**: Iterative agent for intelligent question answering over knowledge graphs
- **`graphRAG_pipeline.py`**: End-to-end pipeline orchestrating the entire workflow
- **`vllm_client.py`**: Client interface for interacting with the vLLM server
- **`prompts.py`**: Prompt templates for various LLM tasks
- **`kg_question.py`**: Question processing utilities
- **`test_graphRAG_contribution.py`**: Validation script to test knowledge graph contribution
- **`performance_benchmark.py`**: Performance measurement and bottleneck analysis script
- **`variable_definitions/`**: Entity definition schemas (default and ontological)
- **`graphRAG_demo.ipynb`**: Interactive demonstration notebook

## Performance Optimization Opportunities

This project presents several areas for acceleration suitable for hackathon optimization:

### Current Bottlenecks

- **Knowledge Graph Construction**: Entity/relation extraction across large document collections (sequential processing)
- **Embedding Generation**: Batch processing of semantic embeddings for graph nodes
- **Graph Traversal**: Iterative exploration and similarity searches during question answering
- **Parallel Processing**: Could improve inference speeds
- **Memory Management**: Inefficient GPU memory utilization during concurrent operations

### Optimization Targets

- **Batch Processing**: Vectorized operations for entity extraction and embedding generation
- **Parallel Graph Construction**: Increasing graph scale and complexity
- **Efficient Vector Search**: GPU-accelerated similarity computations for graph exploration
- **Pipeline Parallelization**: Overlapping data retrieval, processing, and inference stages
- **Model Optimization**: Quantization improvements and memory-efficient attention mechanisms

### Scalability Goals

- Process 100+ research papers simultaneously (currently handles ~10-20)
- Build knowledge graphs with more nodes and embeddings in reasonable time
- Real-time question answering over large knowledge bases
- Handle multiple concurrent users/queries

## Starting the vLLM Server

The `run_vllm.sh` script launches a vLLM API server using the `Qwen/Qwen2.5-7B-Instruct-AWQ` model:

```bash
bash run_vllm.sh
```

The server configuration includes:

- **Model**: Qwen/Qwen2.5-7B-Instruct-AWQ
- **Quantization**: AWQ Marlin for faster inference
- **Max Model Length**: 8192 tokens
- **Port**: 8000 (default)
- **GPU Memory Utilization**: 85%

## Usage

### 1. Interactive Demo

Start a Jupyter environment and open the demonstration notebook:

```bash
jupyter notebook graphRAG_demo.ipynb
```

Follow the cells to see the complete workflow from data retrieval to question answering.

### 2. Command Line Pipeline

Use the end-to-end pipeline script for custom questions:

```bash
python3 graphRAG_pipeline.py "your scientific question here"
```

For research-quality results with full control:

```bash
# Basic research question
python3 kg_question.py --question "What are the mechanisms of Alzheimer's disease?"

# Verbose output to see detailed process
python3 kg_question.py --question "How do mRNA vaccines work?" --verbose

# Custom parameters for comprehensive research
python3 kg_question.py \
  --question "What are the latest treatments for cancer?" \
  --pubmed-limit 15 \
  --pmc-limit 5 \
  --max-iterations 7 \
  --out "cancer_research.gml"
```

This will:

1. Fetch relevant literature based on your question
2. Build a knowledge graph from the retrieved texts
3. Use the iterative agent to explore and answer your question

### 3. Individual Components

You can also use individual components:

```python
from data_retriever import fetch_pubmed_abstracts, search_europe_pmc
from kg_builder import build_graph_from_texts
from graphRAG_agent import IterativeKnowledgeGraphAgent

# Fetch data
abstracts = fetch_pubmed_abstracts("cancer immunotherapy", retmax=10)

# Build knowledge graph
graph = build_graph_from_texts([abstracts], llm_client)

# Question answering
agent = IterativeKnowledgeGraphAgent("graph.gml", vllm_client)
answer = agent.answer_question("How does immunotherapy work against cancer?")
```

### 4. Testing and Validation

Validate that your knowledge graph is contributing unique knowledge beyond the LLM's pre-training:

```bash
python3 test_graphRAG_contribution.py
```

This comprehensive test suite performs four types of validation:

1. **LLM Baseline Test**: Tests the LLM's answers without any graph context
2. **Empty Graph Test**: Tests GraphRAG behavior with an empty knowledge graph
3. **Actual Graph Test**: Tests with your real knowledge graph and analyzes information sources
4. **Fabricated Information Test**: Creates fake but plausible information to verify the system uses graph data

The test provides detailed analysis including:

- Answer length and specificity comparisons
- Confidence scores and information completeness
- Source node examination and content verification
- Relevance scoring of extracted information
- Recommendations for improving graph contribution

### 5. Performance Benchmarking

Measure system performance and identify optimization opportunities using the comprehensive benchmark suite:

```bash
# Run full benchmark suite
python3 performance_benchmark.py

# Quick benchmark with reduced parameters
python3 performance_benchmark.py --quick

# Test specific components
python3 performance_benchmark.py --component data    # Data retrieval only
python3 performance_benchmark.py --component kg     # Knowledge graph construction
python3 performance_benchmark.py --component qa     # Question answering
python3 performance_benchmark.py --component concurrent  # Parallel processing

# Test GPU monitoring
python3 performance_benchmark.py --test-gpu

# Debug mode with system information
python3 performance_benchmark.py --debug
```

The benchmark suite provides:

- **System Resource Monitoring**: Real-time CPU, memory, and GPU utilization tracking
- **Performance Metrics**: Throughput measurements (nodes/sec, papers/sec, chars/sec)
- **Bottleneck Identification**: Pinpoints sequential processing and memory inefficiencies
- **Optimization Targets**: Clear areas for hackathon improvements (GPU utilization, parallel processing)
- **Baseline Measurements**: Before/after comparison capabilities for optimization efforts

```
PSC-CMU-PITT HACKATHON - GRAPHRAG PERFORMANCE BENCHMARK
SYSTEM INFORMATION
CPU cores: 16
RAM: 32.0 GB
GPU detected: NVIDIA GeForce RTX 5060 Ti (15.9GB)

Benchmarking Knowledge Graph Construction (8 texts)
Graph built: 45.12s
120 nodes, 158 edges
2.7 nodes/sec

HACKATHON OPTIMIZATION TARGETS:
   • GPU memory utilization optimization
   • Parallel document processing pipeline
   • Vectorized embedding generation
```

### **Graph Structure**

The knowledge graph contains:

- **Documents**: Original research papers and abstracts
- **Chunks**: Segmented text pieces for processing
- **AtomicFacts**: Minimal, self-contained statements
- **KeyElements**: Important entities and concepts
- ** Embeddings**: 384-dimensional semantic vectors for all nodes
- **Relationships**: HAS_CHUNK, HAS_FACT, HAS_KEY_ELEMENT, NEXT

## Jupyter Notebook

For interactive exploration, use the included notebook:

```bash
jupyter notebook docGraphRAG.ipynb
```

The notebook provides step-by-step examples and allows for custom graph analysis.

## Configuration

### Variable Definitions

The system supports two types of entity definitions:

- **`default_definitions.json`**: Standard biomedical entities
- **`ontological_definitions.json`**: Ontology-based entity definitions

### vLLM Server Options

You can modify `run_vllm.sh` to:

- Change model parameters (memory utilization, max sequences, etc.)
- Use different model variants
