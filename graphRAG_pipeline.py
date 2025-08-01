#!/usr/bin/env python3
"""
Simple end-to-end test script for the GraphRAG pipeline.
Usage: python3 test_custom_question.py "your question here"
"""

import sys
import argparse
import tempfile
import os

from data_retriever import (
    fetch_pubmed_abstracts,
    search_europe_pmc,
    fetch_preprint_abstracts
)
from kg_builder import build_graph_from_texts, save_graph_gml
from vllm_client import VLLMClient
from graphRAG_agent import IterativeKnowledgeGraphAgent


def run_graphrag_pipeline(question: str, use_preprints: bool = True, max_iterations: int = 3):
    """
    Run the complete GraphRAG pipeline for a single question.
    
    Args:
        question: The question to answer
        use_preprints: Whether to include preprint data
        max_iterations: Maximum iterations for the agent
    
    Returns:
        Dict with the complete result
    """
    
    print(f"🔍 Question: {question}")
    print("=" * 60)
    
    # Step 1: Retrieve data
    print("📡 Step 1: Retrieving biomedical data...")
    all_texts = []
    
    # PubMed abstracts
    print("  📄 Fetching PubMed abstracts...")
    try:
        abstracts = fetch_pubmed_abstracts(question, retmax=5)
        if abstracts and abstracts.strip():
            all_texts.append(abstracts)
            print(f"    ✅ Retrieved {len(abstracts)} characters from PubMed")
        else:
            print("    ⚠️  No PubMed abstracts found")
    except Exception as e:
        print(f"    ❌ PubMed error: {e}")
    
    # Europe PMC full-text
    print("  📄 Searching Europe PMC...")
    try:
        hits = search_europe_pmc(question, page_size=3)
        for hit in hits[:2]:
            if hit.get("isOpenAccess") == "Y" and "fullTextUrlList" in hit:
                try:
                    import requests
                    url = hit["fullTextUrlList"]["fullTextUrl"][0]["url"]
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        all_texts.append(response.text[:8000])  # Limit size
                        print(f"    ✅ Retrieved full text from Europe PMC")
                        break
                except Exception:
                    continue
        
        if not any("Europe PMC" in str(print) for print in []):  # Check if we added any
            print("    ⚠️  No open access articles found")
    except Exception as e:
        print(f"    ❌ Europe PMC error: {e}")
    
    # Preprints (optional)
    if use_preprints:
        print("  📄 Fetching preprints...")
        try:
            preprints = fetch_preprint_abstracts(question, retmax=3)
            if preprints:
                preprint_text = ""
                for preprint in preprints:
                    if 'abstract' in preprint:
                        preprint_text += preprint['abstract'] + "\n\n"
                if preprint_text.strip():
                    all_texts.append(preprint_text)
                    print(f"    ✅ Retrieved {len(preprints)} preprint abstracts")
                else:
                    print("    ⚠️  No preprint abstracts found")
            else:
                print("    ⚠️  No preprints found")
        except Exception as e:
            print(f"    ❌ Preprints error: {e}")
    
    if not all_texts:
        print("❌ No data retrieved. Cannot continue.")
        return None
    
    print(f"Total sources: {len(all_texts)}")
    print(f"Total content: {sum(len(text) for text in all_texts)} characters")
    
    # Step 2: Build knowledge graph
    print("\n🔗 Step 2: Building knowledge graph...")
    try:
        llm_client = VLLMClient(schema=None)
        graph = build_graph_from_texts(all_texts, llm=llm_client)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.gml', delete=False)
        graph_path = temp_file.name
        temp_file.close()
        
        save_graph_gml(graph, graph_path)
        
        print(f"    ✅ Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
    except Exception as e:
        print(f"    ❌ Graph construction failed: {e}")
        return None
    
    # Step 3: Answer question
    print("\n🤔 Step 3: Answering question with GraphRAG agent...")
    try:
        client = VLLMClient(schema=None)
        agent = IterativeKnowledgeGraphAgent(
            gml_file_path=graph_path,
            vllm_client=client,
            max_iterations=max_iterations
        )
        
        result = agent.answer_question(question)
        
        print(f"    Answer generated!")
        print(f"    Confidence: {result['final_answer'].confidence:.2f}")
        print(f"    Iterations: {result['iterations_completed']}")
        print(f"    Nodes explored: {len(result['explored_nodes'])}")
        
        # Cleanup
        try:
            os.unlink(graph_path)
        except:
            pass
        
        return result
        
    except Exception as e:
        print(f"    ❌ Question answering failed: {e}")
        # Cleanup
        try:
            os.unlink(graph_path)
        except:
            pass
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test the GraphRAG pipeline with a custom question"
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--no-preprints", action="store_true", 
                       help="Skip preprint data retrieval")
    parser.add_argument("--max-iterations", type=int, default=3,
                       help="Maximum iterations for the agent (default: 3)")
    
    args = parser.parse_args()
    
    # Run the pipeline
    result = run_graphrag_pipeline(
        question=args.question,
        use_preprints=not args.no_preprints,
        max_iterations=args.max_iterations
    )
    
    if result:
        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("-" * 40)
        print(result['final_answer'].answer)
        print("-" * 40)
        print(f"Confidence: {result['final_answer'].confidence:.2f}")
        
        print(f"\nEXPLORATION SUMMARY")
        print(f"• Sources used: {len(result['final_answer'].sources)} nodes")
        print(f"• Information gathered: {len(result['notebook'])} entries")
        print(f"• Exploration iterations: {result['iterations_completed']}")
        
        if result['notebook']:
            print(f"\n🔍 KEY FINDINGS")
            for i, entry in enumerate(sorted(result['notebook'], 
                                           key=lambda x: x.relevance_score, 
                                           reverse=True)[:3], 1):
                print(f"{i}. {entry.information[:100]}... (score: {entry.relevance_score:.2f})")
        
        print("\nPipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
