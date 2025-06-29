# GraphRAG: Graph-Based Retrieval-Augmented Generation for Biomedical Knowledge Access

GraphRAG is an open-source framework that combines a knowledge graph with retrieval-augmented generation to improve large language model (LLM) outputs on complex biomedical queries. The project is built for the PSC/CMU/Pitt Open Hackathon and demonstrates how NVIDIA GPUs can accelerate biomedical data processing and language model workflows.

## Project Goals
- Leverage NVIDIA GPU acceleration for graph-based retrieval and LLM inference
- Provide a reproducible example of retrieval-augmented generation using biomedical datasets
- Offer a foundation for researchers to extend and optimize for large-scale knowledge graphs

## Getting Started
1. **Clone the repository**
```bash
git clone <repo-url>
cd PSC-CMU-Pitt-Hackathon
```

2. **Install dependencies** (requires CUDA-enabled GPU)
```bash
pip install -r requirements.txt
```

3. **Run the example**
```bash
python train.py --epochs 1
```

## Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- PyTorch Geometric (for graph processing)

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


