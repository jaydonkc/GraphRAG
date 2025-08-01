#!/usr/bin/env python3
# kg_question.py - Complete GraphRAG pipeline from query to answer

import argparse
import requests
import os
from pathlib import Path

from data_retriever import (
    fetch_pubmed_abstracts,
    search_europe_pmc,
    fetch_pmc_fulltext,
    fetch_preprint_abstracts
)
from kg_builder import build_graph_from_texts, save_graph_gml
from vllm_client import VLLMClient
from graphRAG_agent import IterativeKnowledgeGraphAgent


def main():
    parser = argparse.ArgumentParser(
        description="Build KG from biomedical sources and run GraphRAG agent"
    )
    parser.add_argument("--question", required=True, 
                       help="Question to answer using biomedical literature")
    parser.add_argument("--out", default="graph_dump.gml",
                       help="Output path for the knowledge graph (default: graph_dump.gml)")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum iterations for the GraphRAG agent (default: 5)")
    parser.add_argument("--pubmed-limit", type=int, default=10,
                       help="Maximum PubMed abstracts to retrieve (default: 10)")
    parser.add_argument("--pmc-limit", type=int, default=5,
                       help="Maximum Europe PMC articles to retrieve (default: 5)")
    parser.add_argument("--preprint-limit", type=int, default=5,
                       help="Maximum preprints to retrieve (default: 5)")
    parser.add_argument("--skip-preprints", action="store_true",
                       help="Skip preprint retrieval")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()

    print(f" Question: {args.question}")
    print("=" * 80)
    
    # 1) Data Retrieval
    print(" STEP 1: RETRIEVING BIOMEDICAL DATA")
    all_texts = []
    
    # PubMed abstracts
    if args.verbose:
        print(f"[retriever] Fetching up to {args.pubmed_limit} PubMed abstracts...")
    else:
        print("   Fetching PubMed abstracts...")
    
    try:
        abstracts = fetch_pubmed_abstracts(args.question, retmax=args.pubmed_limit)
        if abstracts and abstracts.strip():
            all_texts.append(abstracts)
            print(f"     Retrieved {len(abstracts)} characters from {args.pubmed_limit} abstracts")
        else:
            print("      No PubMed abstracts found")
    except Exception as e:
        print(f"     PubMed error: {e}")

    # Europe PMC full-text articles
    if args.verbose:
        print(f"[retriever] Searching Europe PMC for up to {args.pmc_limit} articles...")
    else:
        print("   Searching Europe PMC...")
    
    try:
        hits = search_europe_pmc(args.question, page_size=args.pmc_limit)
        full_texts = []
        
        for i, hit in enumerate(hits[:args.pmc_limit]):
            if hit.get("isOpenAccess") == "Y" and "fullTextUrlList" in hit:
                try:
                    url = hit["fullTextUrlList"]["fullTextUrl"][0]["url"]
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        full_texts.append(response.text)
                        if args.verbose:
                            print(f"    Retrieved full text {i+1}: {hit.get('title', 'Unknown')[:60]}...")
                except Exception as e:
                    if args.verbose:
                        print(f"    Failed to fetch full text {i+1}: {e}")
        
        if full_texts:
            all_texts.extend(full_texts)
            print(f"     Retrieved {len(full_texts)} full-text articles")
        else:
            print("      No open access articles found")
    except Exception as e:
        print(f"     Europe PMC error: {e}")

    # Preprints
    if not args.skip_preprints:
        if args.verbose:
            print(f"[retriever] Fetching up to {args.preprint_limit} preprints...")
        else:
            print("   Fetching preprints...")
        
        try:
            preprints = fetch_preprint_abstracts(args.question, retmax=args.preprint_limit)
            if preprints:
                # Combine preprint abstracts
                preprint_text = ""
                for preprint in preprints:
                    if 'abstract' in preprint:
                        preprint_text += f"Title: {preprint.get('title', 'Unknown')}\n"
                        preprint_text += f"Abstract: {preprint['abstract']}\n\n"
                
                if preprint_text.strip():
                    all_texts.append(preprint_text)
                    print(f"     Retrieved {len(preprints)} preprint abstracts")
                else:
                    print("      No preprint abstracts found")
            else:
                print("      No preprints found")
        except Exception as e:
            print(f"     Preprints error: {e}")

    if not all_texts:
        print(" No data retrieved. Cannot build knowledge graph.")
        return

    print(f" Summary: {len(all_texts)} text sources, {sum(len(t) for t in all_texts):,} total characters")

    # 2) Knowledge Graph Construction
    print(f"\n STEP 2: BUILDING KNOWLEDGE GRAPH")
    
    if args.verbose:
        print("[kg_builder] Initializing vLLMClient for graph construction...")
    else:
        print("   Initializing LLM client...")
    
    try:
        build_client = VLLMClient(schema=None)
        
        if args.verbose:
            print("[kg_builder] Constructing graph from texts...")
        else:
            print("   Building knowledge graph...")
        
        G = build_graph_from_texts(all_texts, llm=build_client)
        
        # Ensure output directory exists
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_graph_gml(G, args.out)
        
        print(f"     Graph saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"     Saved to: {args.out}")
        
        # Show node type distribution
        if args.verbose:
            node_types = {}
            for _, data in G.nodes(data=True):
                labels = data.get('labels', ['Unknown'])
                for label in labels:
                    node_types[label] = node_types.get(label, 0) + 1
            print(f"     Node types: {dict(node_types)}")
            
    except Exception as e:
        print(f"     Graph construction failed: {e}")
        return

    # 3) Question Answering with GraphRAG Agent
    print(f"\n🤔hmm STEP 3: ANSWERING QUESTION WITH GRAPHRAG AGENT")
    
    if args.verbose:
        print("[agent] Initializing vLLMClient for agent...")
    else:
        print("   Initializing agent...")

    try:
        client = VLLMClient(schema=None)
        
        if args.verbose:
            print(f"[agent] Loading IterativeKnowledgeGraphAgent with max_iterations={args.max_iterations}...")
        else:
            print("   Loading GraphRAG agent...")
        
        agent = IterativeKnowledgeGraphAgent(
            gml_file_path=args.out,
            vllm_client=client,
            max_iterations=args.max_iterations
        )

        print(f"   Running iterative exploration...")
        result = agent.answer_question(args.question)

        print(f"     Exploration completed!")
        print(f"     Iterations: {result['iterations_completed']}/{args.max_iterations}")
        print(f"     Nodes explored: {len(result['explored_nodes'])}")
        print(f"     Information gathered: {len(result['notebook'])} entries")
        print(f"     Confidence: {result['final_answer'].confidence:.2f}")

    except Exception as e:
        print(f"     Question answering failed: {e}")
        return

    # 4) Results
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("-" * 40)
    print(result["final_answer"].answer)
    print("-" * 40)
    print(f"Confidence: {result['final_answer'].confidence:.2f}")
    print(f"Sources: {len(result['final_answer'].sources)} nodes")
    print(f"Completeness: {result['final_answer'].information_completeness:.2f}")
    
    if args.verbose and result["notebook"]:
        print(f"\nEXPLORATION NOTEBOOK")
        print("-" * 40)
        for i, entry in enumerate(result["notebook"], 1):
            print(f"{i}. Node {entry.source_node_id}")
            print(f"   Information: {entry.information[:200]}...")
            print(f"   Type: {entry.information_type}, Relevance: {entry.relevance_score:.2f}\n")
    
    print(f"\n Knowledge graph saved to: {args.out}")
    print(" GraphRAG pipeline completed successfully!")


if __name__ == "__main__":
    main()
