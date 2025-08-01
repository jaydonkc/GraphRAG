#!/usr/bin/env python3
"""
Test script to validate that the knowledge graph is contributing unique knowledge
beyond what the LLM already knows from pre-training.
"""

import json
import networkx as nx
from vllm_client import VLLMClient
from graphRAG_agent import IterativeKnowledgeGraphAgent
import tempfile
import os

def test_llm_baseline_knowledge(question: str, client: VLLMClient) -> str:
    """Test what the LLM knows without any graph context"""
    
    prompt = f"""Question: {question}

Please answer this question using only your pre-trained knowledge. Do not make up any specific citations, studies, or research papers. If you're uncertain about specific details, say so.

Answer:"""
    
    response = client(prompt, sampling_params={
        "max_tokens": 500,
        "temperature": 0.1
    })
    
    return response

def test_empty_graph_response(question: str) -> str:
    """Test GraphRAG with an empty graph to see baseline behavior"""
    
    # Create a minimal empty graph
    empty_graph = nx.Graph()
    empty_graph.add_node("empty", labels=["EmptyNode"], text="This is an empty node with no real information.")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.gml', delete=False) as f:
        nx.write_gml(empty_graph, f.name)
        temp_path = f.name
    
    try:
        client = VLLMClient(schema=None)
        agent = IterativeKnowledgeGraphAgent(
            gml_file_path=temp_path,
            vllm_client=client,
            max_iterations=2  # Short exploration
        )
        
        result = agent.answer_question(question)
        return result["final_answer"].answer
    finally:
        os.unlink(temp_path)

def test_specific_graph_content(question: str, graph_path: str) -> dict:
    """Test with actual graph and analyze what information comes from where"""
    
    client = VLLMClient(schema=None)
    agent = IterativeKnowledgeGraphAgent(
        gml_file_path=graph_path,
        vllm_client=client,
        max_iterations=3
    )
    
    result = agent.answer_question(question)
    
    # Analyze the sources
    analysis = {
        "answer": result["final_answer"].answer,
        "confidence": result["final_answer"].confidence,
        "nodes_explored": len(result["explored_nodes"]),
        "information_entries": len(result["notebook"]),
        "source_nodes": [entry.source_node_id for entry in result["notebook"]],
        "information_types": [entry.information_type for entry in result["notebook"]],
        "relevance_scores": [entry.relevance_score for entry in result["notebook"]],
        "extracted_info": [entry.information for entry in result["notebook"]]
    }
    
    return analysis

def examine_node_content(graph_path: str, node_ids: list) -> dict:
    """Examine the actual content of specific nodes to verify information source"""
    
    G = nx.read_gml(graph_path)
    node_contents = {}
    
    for node_id in node_ids:
        if node_id in G:
            node_data = G.nodes[node_id]
            content = {
                "labels": node_data.get("labels", []),
                "text": node_data.get("text", "")[:500] + "..." if len(node_data.get("text", "")) > 500 else node_data.get("text", ""),
                "type": node_data.get("type", "unknown")
            }
            node_contents[node_id] = content
    
    return node_contents

def create_test_with_fabricated_info():
    """Create a test with clearly fabricated information that the LLM couldn't know"""
    
    # Create a graph with fake but plausible medical information
    fake_graph = nx.Graph()
    
    # Add some fake entities and relationships
    fake_graph.add_node("fake_study_1", 
                        labels=["Document"], 
                        type="Document",
                        text="A recent study by Dr. Zephyr Williams at the Institute of Advanced Metabolics found that the protein XYLOSE-7 plays a crucial role in insulin resistance. The study followed 2,847 patients over 5 years and found that elevated XYLOSE-7 levels were present in 89% of patients who developed type 2 diabetes, compared to only 12% in the control group.")
    
    fake_graph.add_node("fake_protein_1",
                        labels=["KeyElement"],
                        type="KeyElement", 
                        text="XYLOSE-7 protein")
    
    fake_graph.add_node("fake_mechanism_1",
                        labels=["AtomicFact"],
                        type="AtomicFact",
                        text="XYLOSE-7 interferes with glucose uptake in muscle cells by binding to GLUT4 transporters")
    
    # Add edges
    fake_graph.add_edge("fake_study_1", "fake_protein_1", relation="MENTIONS")
    fake_graph.add_edge("fake_study_1", "fake_mechanism_1", relation="HAS_FINDING")
    fake_graph.add_edge("fake_protein_1", "fake_mechanism_1", relation="CAUSES")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.gml', delete=False) as f:
        nx.write_gml(fake_graph, f.name)
        return f.name

def run_validation_tests(question: str, graph_path: str):
    """Run comprehensive validation tests"""
    
    print("TESTING KNOWLEDGE GRAPH CONTRIBUTION")
    print("=" * 60)
    print(f"Question: {question}")
    print("\n")
    
    client = VLLMClient(schema=None)
    
    # Test 1: LLM baseline knowledge
    print("TESTING LLM BASELINE (No Graph)")
    print("-" * 40)
    try:
        baseline_answer = test_llm_baseline_knowledge(question, client)
        print("Baseline Answer:")
        print(baseline_answer)
        print(f"Length: {len(baseline_answer)} characters")
    except Exception as e:
        print(f"Error in baseline test: {e}")
        baseline_answer = "Error occurred"
    
    print("\n")
    
    # Test 2: Empty graph
    print("TESTING WITH EMPTY GRAPH")
    print("-" * 40)
    try:
        empty_graph_answer = test_empty_graph_response(question)
        print("Empty Graph Answer:")
        print(empty_graph_answer)
        print(f"Length: {len(empty_graph_answer)} characters")
    except Exception as e:
        print(f"Error in empty graph test: {e}")
        empty_graph_answer = "Error occurred"
    
    print("\n")
    
    # Test 3: Actual graph
    print("TESTING WITH ACTUAL GRAPH")
    print("-" * 40)
    try:
        graph_analysis = test_specific_graph_content(question, graph_path)
        print("Graph-Enhanced Answer:")
        print(graph_analysis["answer"])
        print(f"Length: {len(graph_analysis['answer'])} characters")
        print(f"Confidence: {graph_analysis['confidence']}")
        print(f"Nodes explored: {graph_analysis['nodes_explored']}")
        print(f"Information entries: {graph_analysis['information_entries']}")
        
        if graph_analysis["source_nodes"]:
            print(f"\nSource nodes used: {graph_analysis['source_nodes']}")
            
            # Examine the actual node content
            print("\nEXAMINING SOURCE NODE CONTENT:")
            node_contents = examine_node_content(graph_path, graph_analysis["source_nodes"][:3])
            for i, (node_id, content) in enumerate(node_contents.items(), 1):
                print(f"\nNode {i} ({node_id}):")
                print(f"  Type: {content['type']}")
                print(f"  Labels: {content['labels']}")
                print(f"  Content: {content['text'][:200]}...")
        
        print("\n🔍 EXTRACTED INFORMATION FROM GRAPH:")
        for i, info in enumerate(graph_analysis["extracted_info"], 1):
            relevance = graph_analysis["relevance_scores"][i-1]
            print(f"{i}. (Relevance: {relevance:.2f}) {info[:150]}...")
            
    except Exception as e:
        print(f"Error in graph test: {e}")
        graph_analysis = None
    
    print("\n")
    
    # Test 4: Fabricated information test
    print("TESTING WITH FABRICATED INFORMATION")
    print("-" * 40)
    print("Creating graph with fake but plausible medical information...")
    
    fake_graph_path = create_test_with_fabricated_info()
    try:
        fake_question = "What role does XYLOSE-7 protein play in insulin resistance?"
        fake_analysis = test_specific_graph_content(fake_question, fake_graph_path)
        
        print(f"Question: {fake_question}")
        print("Answer with fabricated info:")
        print(fake_analysis["answer"])
        print(f"Confidence: {fake_analysis['confidence']}")
        print(f"Used {fake_analysis['information_entries']} pieces of fabricated information")
        
        # Check if the fake information appears in the answer
        if "XYLOSE-7" in fake_analysis["answer"]:
            print("SUCCESS: The system used fabricated graph information!")
        else:
            print("CONCERN: The system may not be using graph information effectively")
            
    except Exception as e:
        print(f"Error in fabricated test: {e}")
    finally:
        os.unlink(fake_graph_path)
    
    print("\n")
    
    # Analysis and conclusions
    print("ANALYSIS & CONCLUSIONS")
    print("-" * 40)
    
    if graph_analysis and baseline_answer != "Error occurred":
        # Compare lengths and content
        baseline_len = len(baseline_answer)
        graph_len = len(graph_analysis["answer"])
        
        print(f"Answer length comparison:")
        print(f"  Baseline (no graph): {baseline_len} chars")
        print(f"  With graph: {graph_len} chars")
        print(f"  Difference: {graph_len - baseline_len:+d} chars")
        
        # Check for specific details that might come from graph
        if graph_analysis["information_entries"] > 0:
            print(f"\nGraph provided {graph_analysis['information_entries']} specific information entries")
            print(f"Explored {graph_analysis['nodes_explored']} nodes in the knowledge graph")
            print(f"Average relevance score: {sum(graph_analysis['relevance_scores'])/len(graph_analysis['relevance_scores']):.2f}")
        else:
            print("\nWarning: No information was extracted from the graph")
    
    print("\n💡 RECOMMENDATIONS:")
    print("• Compare the specificity and citations in graph vs baseline answers")
    print("• Look for unique terms, numbers, or studies mentioned only in graph answer")
    print("• Check if graph answer has higher confidence when graph contains relevant info")
    print("• Verify that fabricated information test shows clear graph usage")

def main():
    """Main validation function"""
    
    # Check if we have a graph file
    graph_files = ["graph_dump.gml", "diabetes_graph.gml"]
    graph_file = None
    
    for gf in graph_files:
        if os.path.exists(gf):
            graph_file = gf
            break
    
    if not graph_file:
        print("No graph file found. Please run the pipeline first to create a graph.")
        print("Available tests: Create a graph with:")
        print("  python3 test_custom_question.py 'your question'")
        return
    
    print(f"Using graph file: {graph_file}")
    
    # Test questions - mix of general and specific
    test_questions = [
        "What are the main risk factors for type 2 diabetes?",
        "How does insulin resistance develop?",
        "What is the relationship between obesity and diabetes?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_questions)}")
        print('='*80)
        run_validation_tests(question, graph_file)
        
        if i < len(test_questions):
            input("\nPress Enter to continue to next test...")

if __name__ == "__main__":
    main()
