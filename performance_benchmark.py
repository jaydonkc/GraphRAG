#!/usr/bin/env python3
"""
Performance benchmark script to measure current system performance and identify bottlenecks.
This script helps establish baseline metrics and guide optimization efforts for the hackathon.
"""

import time
import psutil
import gc
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import networkx as nx

# Try to import GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPU_AVAILABLE = False
    print("GPUtil not available. Install with: pip install GPUtil")

from data_retriever import fetch_pubmed_abstracts, search_europe_pmc, fetch_preprint_abstracts
from kg_builder import build_graph_from_texts
from vllm_client import VLLMClient
from graphRAG_agent import IterativeKnowledgeGraphAgent


def test_gpu_availability():
    """Test if GPU monitoring is working properly"""
    if not GPU_AVAILABLE or GPUtil is None:
        return False, "GPUtil not imported"
    
    try:
        gpus = GPUtil.getGPUs()
        if not gpus or len(gpus) == 0:
            return False, "No GPUs detected"
        
        # Test basic GPU info access
        gpu = gpus[0]
        name = gpu.name
        memory = gpu.memoryTotal
        load = gpu.load
        
        return True, f"GPU detected: {name} ({memory/1024:.1f}GB)"
        
    except Exception as e:
        # Note: We can't modify the global here due to scope issues
        return False, f"GPU access error: {e}"


def print_system_info():
    """Print system information and GPU status"""
    print("SYSTEM INFORMATION")
    print("-" * 40)
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    gpu_working, gpu_msg = test_gpu_availability()
    if gpu_working:
        print(f"{gpu_msg}")
    else:
        print(f"GPU monitoring: {gpu_msg}")
    
    print()


class PerformanceMonitor:
    """Monitor system resources during benchmarks"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_gb': [],
            'gpu_utilization': [],
            'gpu_memory_used': [],
            'timestamps': []
        }
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.metrics = {key: [] for key in self.metrics.keys()}
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.metrics.copy()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # CPU and Memory
                self.metrics['cpu_percent'].append(psutil.cpu_percent())
                memory = psutil.virtual_memory()
                self.metrics['memory_percent'].append(memory.percent)
                self.metrics['memory_used_gb'].append(memory.used / (1024**3))
                
                # GPU metrics if available
                if GPU_AVAILABLE and GPUtil is not None:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus and len(gpus) > 0:
                            gpu = gpus[0]  # Use first GPU
                            self.metrics['gpu_utilization'].append(gpu.load * 100)
                            self.metrics['gpu_memory_used'].append(gpu.memoryUsed)
                        else:
                            # No GPUs found
                            self.metrics['gpu_utilization'].append(0)
                            self.metrics['gpu_memory_used'].append(0)
                    except Exception as gpu_error:
                        # GPUtil error - disable for this session
                        print(f"⚠️ GPU monitoring disabled due to error: {gpu_error}")
                        self.metrics['gpu_utilization'].append(0)
                        self.metrics['gpu_memory_used'].append(0)
                else:
                    self.metrics['gpu_utilization'].append(0)
                    self.metrics['gpu_memory_used'].append(0)
                
                self.metrics['timestamps'].append(time.time())
                time.sleep(0.5)  # Sample every 500ms
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                break


def benchmark_data_retrieval(query: str = "machine learning cancer", num_papers: int = 20) -> Dict[str, Any]:
    """Benchmark data retrieval from multiple sources"""
    print(f"Benchmarking Data Retrieval ({num_papers} papers)")
    print("-" * 50)
    
    results = {}
    monitor = PerformanceMonitor()
    
    # Benchmark PubMed
    print("Testing PubMed retrieval...")
    monitor.start_monitoring()
    start_time = time.time()
    
    try:
        pubmed_data = fetch_pubmed_abstracts(query, retmax=num_papers)
        pubmed_time = time.time() - start_time
        pubmed_size = len(pubmed_data) if pubmed_data else 0
        
        results['pubmed'] = {
            'time_seconds': pubmed_time,
            'data_size_chars': pubmed_size,
            'papers_retrieved': num_papers,
            'throughput_chars_per_sec': pubmed_size / pubmed_time if pubmed_time > 0 else 0
        }
        print(f"✅ PubMed: {pubmed_time:.2f}s, {pubmed_size:,} chars")
        
    except Exception as e:
        print(f"❌ PubMed failed: {e}")
        results['pubmed'] = {'error': str(e)}
        pubmed_metrics = monitor.stop_monitoring()
    else:
        pubmed_metrics = monitor.stop_monitoring()
        if 'error' not in results['pubmed']:
            results['pubmed']['resource_metrics'] = pubmed_metrics
    
    # Benchmark Europe PMC
    print("Testing Europe PMC retrieval...")
    monitor.start_monitoring()
    start_time = time.time()
    
    try:
        pmc_data = search_europe_pmc(query, page_size=min(num_papers, 25))  # API limit
        pmc_time = time.time() - start_time
        pmc_papers = len(pmc_data)
        
        results['europe_pmc'] = {
            'time_seconds': pmc_time,
            'papers_retrieved': pmc_papers,
            'throughput_papers_per_sec': pmc_papers / pmc_time if pmc_time > 0 else 0
        }
        print(f"✅ Europe PMC: {pmc_time:.2f}s, {pmc_papers} papers")
        
    except Exception as e:
        print(f"❌ Europe PMC failed: {e}")
        results['europe_pmc'] = {'error': str(e)}
        pmc_metrics = monitor.stop_monitoring()
    else:
        pmc_metrics = monitor.stop_monitoring()
        if 'error' not in results['europe_pmc']:
            results['europe_pmc']['resource_metrics'] = pmc_metrics
    
    # Benchmark Preprints
    print("Testing preprint retrieval...")
    monitor.start_monitoring()
    start_time = time.time()
    
    try:
        preprint_data = fetch_preprint_abstracts(query, retmax=min(num_papers, 10))
        preprint_time = time.time() - start_time
        preprint_papers = len(preprint_data)
        
        results['preprints'] = {
            'time_seconds': preprint_time,
            'papers_retrieved': preprint_papers,
            'throughput_papers_per_sec': preprint_papers / preprint_time if preprint_time > 0 else 0
        }
        print(f"✅ Preprints: {preprint_time:.2f}s, {preprint_papers} papers")
        
    except Exception as e:
        print(f"❌ Preprints failed: {e}")
        results['preprints'] = {'error': str(e)}
        preprint_metrics = monitor.stop_monitoring()
    else:
        preprint_metrics = monitor.stop_monitoring()
        if 'error' not in results['preprints']:
            results['preprints']['resource_metrics'] = preprint_metrics
    
    return results


def benchmark_knowledge_graph_construction(texts: List[str], use_llm: bool = True) -> Dict[str, Any]:
    """Benchmark knowledge graph construction from texts"""
    print(f"📊 Benchmarking Knowledge Graph Construction ({len(texts)} texts)")
    print("-" * 50)
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    start_time = time.time()
    total_chars = sum(len(text) for text in texts)
    
    try:
        if use_llm:
            llm_client = VLLMClient(schema=None)
            print("Building graph with LLM extraction...")
        else:
            llm_client = None
            print("Building graph without LLM (baseline)...")
        
        # Build the graph
        graph = build_graph_from_texts(texts, llm=llm_client)
        
        construction_time = time.time() - start_time
        metrics = monitor.stop_monitoring()
        
        # Analyze graph structure
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        # Node type analysis
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        results = {
            'construction_time_seconds': construction_time,
            'total_input_chars': total_chars,
            'throughput_chars_per_sec': total_chars / construction_time if construction_time > 0 else 0,
            'graph_stats': {
                'nodes': num_nodes,
                'edges': num_edges,
                'node_types': node_types,
                'average_degree': (2 * num_edges) / num_nodes if num_nodes > 0 else 0
            },
            'performance_metrics': {
                'chars_per_node': total_chars / num_nodes if num_nodes > 0 else 0,
                'nodes_per_second': num_nodes / construction_time if construction_time > 0 else 0
            },
            'resource_metrics': metrics
        }
        
        print(f"✅ Graph built: {construction_time:.2f}s")
        print(f"   📈 {num_nodes:,} nodes, {num_edges:,} edges")
        print(f"   🏃 {results['performance_metrics']['nodes_per_second']:.1f} nodes/sec")
        
        return results
        
    except Exception as e:
        metrics = monitor.stop_monitoring()
        print(f"❌ Graph construction failed: {e}")
        return {
            'error': str(e),
            'construction_time_seconds': time.time() - start_time,
            'resource_metrics': metrics
        }


def benchmark_question_answering(graph_path: str, questions: List[str]) -> Dict[str, Any]:
    """Benchmark question answering performance"""
    print(f"📊 Benchmarking Question Answering ({len(questions)} questions)")
    print("-" * 50)
    
    if not os.path.exists(graph_path):
        return {'error': f'Graph file not found: {graph_path}'}
    
    try:
        # Initialize the agent
        client = VLLMClient(schema=None)
        agent = IterativeKnowledgeGraphAgent(
            gml_file_path=graph_path,
            vllm_client=client,
            max_iterations=3  # Moderate exploration
        )
        
        results = {
            'questions': [],
            'total_time': 0,
            'average_time_per_question': 0,
            'resource_metrics': {}
        }
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        total_start = time.time()
        
        for i, question in enumerate(questions, 1):
            print(f"Question {i}/{len(questions)}: {question[:60]}...")
            
            q_start = time.time()
            try:
                result = agent.answer_question(question)
                q_time = time.time() - q_start
                
                question_result = {
                    'question': question,
                    'time_seconds': q_time,
                    'confidence': result['final_answer'].confidence,
                    'nodes_explored': len(result['explored_nodes']),
                    'information_entries': len(result['notebook']),
                    'answer_length': len(result['final_answer'].answer),
                    'avg_relevance': sum(entry.relevance_score for entry in result['notebook']) / len(result['notebook']) if result['notebook'] else 0
                }
                
                print(f"   ✅ {q_time:.2f}s, confidence: {question_result['confidence']:.2f}")
                
            except Exception as e:
                q_time = time.time() - q_start
                question_result = {
                    'question': question,
                    'time_seconds': q_time,
                    'error': str(e)
                }
                print(f"   ❌ {q_time:.2f}s, error: {str(e)[:50]}...")
            
            results['questions'].append(question_result)
        
        total_time = time.time() - total_start
        metrics = monitor.stop_monitoring()
        
        results['total_time'] = total_time
        results['average_time_per_question'] = total_time / len(questions)
        results['resource_metrics'] = metrics
        
        print(f"\nOverall: {total_time:.2f}s total, {results['average_time_per_question']:.2f}s avg per question")
        
        return results
        
    except Exception as e:
        return {'error': f'Question answering setup failed: {e}'}


def benchmark_concurrent_processing(texts: List[str], max_workers: int = 4) -> Dict[str, Any]:
    """Benchmark concurrent vs sequential processing"""
    print(f"Benchmarking Concurrent Processing (max_workers={max_workers})")
    print("-" * 50)
    
    # Sequential processing
    print("Testing sequential processing...")
    start_time = time.time()
    sequential_graphs = []
    
    for i, text in enumerate(texts):
        try:
            graph = build_graph_from_texts([text], llm=None)  # No LLM for speed
            sequential_graphs.append(graph)
        except Exception as e:
            print(f"Sequential error on text {i}: {e}")
    
    sequential_time = time.time() - start_time
    
    # Concurrent processing
    print("Testing concurrent processing...")
    start_time = time.time()
    concurrent_graphs = []
    
    def process_text(text):
        try:
            return build_graph_from_texts([text], llm=None)
        except Exception as e:
            print(f"Concurrent error: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        concurrent_graphs = list(executor.map(process_text, texts))
    
    concurrent_time = time.time() - start_time
    
    # Analysis
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
    efficiency = speedup / max_workers
    
    results = {
        'sequential_time': sequential_time,
        'concurrent_time': concurrent_time,
        'speedup': speedup,
        'efficiency': efficiency,
        'max_workers': max_workers,
        'texts_processed': len(texts),
        'successful_sequential': len([g for g in sequential_graphs if g is not None]),
        'successful_concurrent': len([g for g in concurrent_graphs if g is not None])
    }
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2f}")
    
    return results


def create_test_documents(num_docs: int = 10) -> List[str]:
    """Create test documents for benchmarking"""
    base_text = """
    Machine learning approaches in cancer research have shown significant promise for improving 
    diagnostic accuracy and treatment outcomes. Deep learning models, particularly convolutional 
    neural networks, have been successfully applied to medical imaging tasks including tumor 
    detection and classification. Recent studies demonstrate that ensemble methods combining 
    multiple algorithms can achieve superior performance compared to individual models.
    
    The integration of multi-omics data, including genomic, transcriptomic, and proteomic 
    information, presents new opportunities for personalized medicine. Graph neural networks 
    have emerged as powerful tools for modeling biological networks and pathway interactions.
    Natural language processing techniques are being used to extract knowledge from vast 
    amounts of biomedical literature, enabling researchers to discover novel associations 
    and generate hypotheses for experimental validation.
    """
    
    # Create variations of the base text
    documents = []
    variations = [
        "cancer treatment", "tumor detection", "medical imaging", "genomic analysis",
        "drug discovery", "biomarker identification", "clinical trials", "precision medicine",
        "immunotherapy", "radiotherapy"
    ]
    
    for i in range(num_docs):
        variation = variations[i % len(variations)]
        doc = base_text.replace("cancer research", f"{variation} research")
        doc = f"Document {i+1}: {doc}"
        documents.append(doc)
    
    return documents


def save_benchmark_results(results: Dict[str, Any], filename: Optional[str] = None):
    """Save benchmark results to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
    
    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"Results saved to: {filename}")
    return filename


def run_comprehensive_benchmark():
    """Run all benchmarks and generate comprehensive report"""
    print("PSC-CMU-PITT HACKATHON - GRAPHRAG PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Print system info first
    print_system_info()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': GPU_AVAILABLE
        }
    }
    
    # Add GPU info if available
    gpu_working, gpu_msg = test_gpu_availability()
    if gpu_working and GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus and len(gpus) > 0:
                gpu = gpus[0]
                results['system_info']['gpu_name'] = gpu.name
                results['system_info']['gpu_memory_gb'] = gpu.memoryTotal / 1024
                results['system_info']['gpu_driver'] = gpu.driver if hasattr(gpu, 'driver') else 'unknown'
        except Exception as e:
            results['system_info']['gpu_error'] = str(e)
    else:
        results['system_info']['gpu_status'] = gpu_msg
    
    # 1. Data Retrieval Benchmark
    try:
        print("\n" + "="*80)
        results['data_retrieval'] = benchmark_data_retrieval(
            query="machine learning cancer treatment", 
            num_papers=15
        )
    except Exception as e:
        print(f"Data retrieval benchmark failed: {e}")
        results['data_retrieval'] = {'error': str(e)}
    
    # 2. Knowledge Graph Construction Benchmark
    try:
        print("\n" + "="*80)
        test_docs = create_test_documents(8)
        results['kg_construction'] = benchmark_knowledge_graph_construction(test_docs, use_llm=False)
        
        # Save graph for next test
        if 'error' not in results['kg_construction']:
            graph = build_graph_from_texts(test_docs, llm=None)
            temp_graph_path = "temp_benchmark_graph.gml"
            nx.write_gml(graph, temp_graph_path)
        
    except Exception as e:
        print(f"KG construction benchmark failed: {e}")
        results['kg_construction'] = {'error': str(e)}
    
    # 3. Question Answering Benchmark
    try:
        print("\n" + "="*80)
        if os.path.exists("temp_benchmark_graph.gml"):
            test_questions = [
                "What are the main applications of machine learning in cancer research?",
                "How do deep learning models help with medical imaging?",
                "What is the role of multi-omics data in personalized medicine?"
            ]
            results['question_answering'] = benchmark_question_answering(
                "temp_benchmark_graph.gml", 
                test_questions
            )
        else:
            results['question_answering'] = {'error': 'No graph available for testing'}
            
    except Exception as e:
        print(f"Question answering benchmark failed: {e}")
        results['question_answering'] = {'error': str(e)}
    
    # 4. Concurrent Processing Benchmark
    try:
        print("\n" + "="*80)
        test_docs_small = create_test_documents(6)
        results['concurrent_processing'] = benchmark_concurrent_processing(test_docs_small, max_workers=4)
        
    except Exception as e:
        print(f"Concurrent processing benchmark failed: {e}")
        results['concurrent_processing'] = {'error': str(e)}
    
    # Clean up
    if os.path.exists("temp_benchmark_graph.gml"):
        os.remove("temp_benchmark_graph.gml")
    
    # Save results
    results_file = save_benchmark_results(results)
    
    # Generate summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    if 'data_retrieval' in results and 'error' not in results['data_retrieval']:
        print("Data Retrieval:")
        for source, data in results['data_retrieval'].items():
            if isinstance(data, dict) and 'time_seconds' in data:
                print(f"   {source}: {data['time_seconds']:.2f}s")
    
    if 'kg_construction' in results and 'error' not in results['kg_construction']:
        kg_data = results['kg_construction']
        print(f"  Knowledge Graph Construction:")
        print(f"  Time: {kg_data['construction_time_seconds']:.2f}s")
        # Safe access to nested dict with type checking
        if isinstance(kg_data, dict) and 'performance_metrics' in kg_data:
            perf_metrics = kg_data['performance_metrics']
            if isinstance(perf_metrics, dict) and 'nodes_per_second' in perf_metrics:
                nodes_per_sec = perf_metrics['nodes_per_second']
                print(f"  Throughput: {nodes_per_sec:.1f} nodes/sec")
    
    if 'question_answering' in results and 'error' not in results['question_answering']:
        qa_data = results['question_answering']
        print(f"   Question Answering:")
        print(f"   Average time: {qa_data['average_time_per_question']:.2f}s per question")
    
    if 'concurrent_processing' in results and 'error' not in results['concurrent_processing']:
        conc_data = results['concurrent_processing']
        print(f"⚡ Concurrent Processing:")
        print(f"   Speedup: {conc_data['speedup']:.2f}x")
        print(f"   Efficiency: {conc_data['efficiency']:.2f}")
    
    print(f"\n Full results: {results_file}")
    print("\n HACKATHON OPTIMIZATION TARGETS:")
    print("   • GPU memory utilization optimization")
    print("   • Parallel document processing pipeline") 
    print("   • Vectorized embedding generation")
    print("   • Efficient graph traversal algorithms")
    print("   • Multi-threading for data retrieval")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG Performance Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with reduced parameters")
    parser.add_argument("--component", choices=["data", "kg", "qa", "concurrent"], help="Run specific component benchmark")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--test-gpu", action="store_true", help="Test GPU monitoring only")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra logging")
    
    args = parser.parse_args()
    
    if args.debug:
        print("DEBUG MODE ENABLED")
        print_system_info()
        
    if args.test_gpu:
        print("TESTING GPU MONITORING")
        print("-" * 30)
        
        gpu_working, gpu_msg = test_gpu_availability()
        print(f"GPU Status: {gpu_msg}")
        
        if gpu_working:
            print("Testing GPU monitoring loop...")
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            time.sleep(2)  # Monitor for 2 seconds
            metrics = monitor.stop_monitoring()
            
            print(f"Collected {len(metrics['gpu_utilization'])} GPU samples")
            if metrics['gpu_utilization']:
                avg_util = sum(metrics['gpu_utilization']) / len(metrics['gpu_utilization'])
                avg_mem = sum(metrics['gpu_memory_used']) / len(metrics['gpu_memory_used'])
                print(f"Average GPU utilization: {avg_util:.1f}%")
                print(f"Average GPU memory used: {avg_mem:.0f} MB")
        
        exit(0)  # Exit after GPU test
    
    if args.component:
        # Run specific component
        if args.component == "data":
            results = benchmark_data_retrieval(num_papers=5 if args.quick else 15)
        elif args.component == "kg":
            docs = create_test_documents(3 if args.quick else 8)
            results = benchmark_knowledge_graph_construction(docs, use_llm=False)
        elif args.component == "concurrent":
            docs = create_test_documents(3 if args.quick else 6)
            results = benchmark_concurrent_processing(docs, max_workers=2 if args.quick else 4)
        else:
            print("QA benchmark requires full setup")
            results = {}
        
        if args.output:
            save_benchmark_results(results, args.output)
        else:
            print(json.dumps(results, indent=2, default=str))
    else:
        # Run comprehensive benchmark
        run_comprehensive_benchmark()
