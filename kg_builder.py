# src/kg_builder.py
import networkx as nx
import hashlib
import json
from typing import Optional

from vllm_client import VLLMClient
import prompts   # your GraphReader prompt templates

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    EMBEDDINGS_AVAILABLE = False

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def build_graph_from_texts(texts: list[str], llm: Optional[VLLMClient] = None) -> nx.Graph:
    """
    texts: list of raw text strings (abstracts, full‑text, preprints, etc)
    llm:  instance of your VLLMClient
    """
    if llm is None:
        # Create a default VLLMClient if none provided
        llm = VLLMClient(schema=None)
    
    G = nx.Graph()

    # 1) Add a document node for each source
    for txt in texts:
        doc_id = md5(txt[:100])
        G.add_node(doc_id, labels=["Document"], type="Document", text=txt)

        # 2) Chunk the document
        chunks = [txt[i : i+2000] for i in range(0, len(txt), 2000)]
        for idx, chunk in enumerate(chunks):
            cid = md5(chunk)
            G.add_node(cid, labels=["Chunk"], type="Chunk", text=chunk, index=idx)
            G.add_edge(doc_id, cid, relation="HAS_CHUNK")
            if idx > 0:
                prev_cid = md5(chunks[idx-1])
                G.add_edge(prev_cid, cid, relation="NEXT")

            # 3a) Extract Atomic Facts from this chunk
            atomic_prompt = prompts.atomic_fact_prompt.format(chunk=chunk)
            atomic_resp = llm(atomic_prompt, sampling_params={
                "max_tokens": 500, 
                "temperature": 0.1, 
                "stop": ["--", "\n\n#"]
            })
            try:
                # Clean the response - sometimes LLMs add extra text
                resp_clean = atomic_resp.strip()
                if resp_clean.startswith("```json"):
                    resp_clean = resp_clean[7:]
                if resp_clean.endswith("```"):
                    resp_clean = resp_clean[:-3]
                resp_clean = resp_clean.strip()
                
                atomic_out = json.loads(resp_clean)
                facts = atomic_out.get("atomic_facts", [])
            except json.JSONDecodeError as e:
                facts = []
                print(f"[WARN] failed to parse atomic facts for chunk {cid}: {str(e)[:100]}...")
                print(f"[DEBUG] Response was: {atomic_resp[:200]}...")

            for fact in facts:
                # fact is already a string from the JSON format: {"atomic_facts": ["fact1","fact2",...]}
                if isinstance(fact, str):
                    fact_text = fact
                else:
                    # Handle case where fact might be a dict for backward compatibility
                    fact_text = fact.get("fact") or fact.get("text") or str(fact)
                    
                fact_id = md5(fact_text)
                G.add_node(fact_id, labels=["AtomicFact"], type="AtomicFact", text=fact_text)
                G.add_edge(cid, fact_id, relation="HAS_FACT")

                # 3b) Extract Key Elements for each fact
                key_prompt = prompts.key_element_prompt.format(fact=fact_text)
                key_resp = llm(key_prompt, sampling_params={
                    "max_tokens": 200, 
                    "temperature": 0.1, 
                    "stop": ["--", "\n\n#"]
                })
                try:
                    # Clean the response
                    resp_clean = key_resp.strip()
                    if resp_clean.startswith("```json"):
                        resp_clean = resp_clean[7:]
                    if resp_clean.endswith("```"):
                        resp_clean = resp_clean[:-3]
                    resp_clean = resp_clean.strip()
                    
                    key_out = json.loads(resp_clean)
                    key_elts = key_out.get("key_elements", [])
                except json.JSONDecodeError as e:
                    key_elts = []
                    print(f"[WARN] failed to parse key elements for fact {fact_id}: {str(e)[:100]}...")

                for elt in key_elts:
                    # elt is already a string from the JSON format: {"key_elements": ["entity1","entity2",...]}
                    if isinstance(elt, str):
                        elt_text = elt
                    else:
                        elt_text = str(elt)
                        
                    eid = md5(elt_text)
                    G.add_node(eid, labels=["KeyElement"], type="KeyElement", text=elt_text)
                    G.add_edge(fact_id, eid, relation="HAS_KEY_ELEMENT")

    return G

def save_graph_gml(G: nx.Graph, path: str):
    nx.write_gml(G, path)
    print(f"[kg_builder] Saved GML to {path}")

def add_embeddings_to_graph(G: nx.Graph, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> nx.Graph:
    """
    Add embeddings to all nodes in the knowledge graph.
    This can be called after building the graph to enhance retrieval.
    """
    if not EMBEDDINGS_AVAILABLE or SentenceTransformer is None:
        print("[WARN] sentence-transformers not available. Skipping embedding generation.")
        return G
    
    print(f"Adding embeddings to {G.number_of_nodes()} nodes using {embedding_model}")
    embedder = SentenceTransformer(embedding_model)
    
    # Collect all text content for batch processing
    node_texts = []
    node_ids = []
    
    for node_id, node_data in G.nodes(data=True):
        # Extract text content from the node
        text_content = ""
        for field in ['text', 'description', 'summary']:
            if field in node_data and node_data[field]:
                text_content += " " + str(node_data[field])
        
        # If no text content, use labels or type
        if not text_content.strip():
            labels = node_data.get('labels', [])
            node_type = node_data.get('type', '')
            text_content = " ".join(labels) + " " + node_type
        
        node_texts.append(text_content.strip() or "empty_node")
        node_ids.append(node_id)
    
    # Generate embeddings in batches
    batch_size = 32
    for i in range(0, len(node_texts), batch_size):
        batch_texts = node_texts[i:i + batch_size]
        batch_ids = node_ids[i:i + batch_size]
        
        embeddings = embedder.encode(batch_texts)
        
        for j, node_id in enumerate(batch_ids):
            G.nodes[node_id]['embedding'] = embeddings[j].tolist()
        
        if len(node_texts) > batch_size:
            print(f"  Processed batch {i//batch_size + 1}/{(len(node_texts)-1)//batch_size + 1}")
    
    print("✅ Embeddings added to all nodes")
    return G
