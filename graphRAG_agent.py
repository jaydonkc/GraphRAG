from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Set
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import json


class ReasoningStep(BaseModel):
    step: str = Field(..., description="A reasoning step in the planning process")
    required_info: List[str] = Field(..., description="Types of information needed for this step")

class QueryPlan(BaseModel):
    reasoning_steps: List[ReasoningStep] = Field(..., description="Step-by-step plan to answer the query")
    key_concepts: List[str] = Field(..., description="Key concepts that need to be found in the knowledge graph")
    search_strategy: str = Field(..., description="Strategy for searching the knowledge graph")
    expected_answer_type: str = Field(..., description="What type of answer is expected (causal, descriptive, comparative, etc.)")

class NotebookEntry(BaseModel):
    source_node_id: str = Field(..., description="ID of the node this information came from")
    information: str = Field(..., description="Key information extracted from the node")
    relevance_score: float = Field(..., description="How relevant this information is (0-1)")
    information_type: str = Field(..., description="Type of information (causal, descriptive, statistical, etc.)")

class ExplorationDecision(BaseModel):
    should_continue: bool = Field(..., description="Whether to continue exploring")
    reasoning: str = Field(..., description="Reasoning for the decision")
    next_nodes_to_explore: List[str] = Field(default=[], description="Specific node IDs to explore next")
    exploration_strategy: str = Field(..., description="How to explore next (neighbors, keywords, specific_nodes)")
    information_gaps: List[str] = Field(default=[], description="What information is still needed")

class FinalAnswer(BaseModel):
    reasoning_steps: List[str] = Field(..., description="Step-by-step reasoning using gathered information")
    answer: str = Field(..., description="Final comprehensive answer to the question")
    confidence: float = Field(..., description="Confidence score (0-1) in the answer")
    sources: List[str] = Field(..., description="Node IDs used as sources for the answer")
    information_completeness: float = Field(..., description="How complete the gathered information is (0-1)")


class IterativeKnowledgeGraphAgent:
    def __init__(self, gml_file_path: str, vllm_client, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ", max_iterations: int = 5, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the iterative Knowledge Graph QA agent
        
            tokenizer_name: Name of the tokenizer to use
            max_iterations: Maximum number of exploration iterations
            embedding_model: Name of the sentence transformer model for embeddings
        """
        self.graph = nx.read_gml(gml_file_path)
        self.vllm_client = vllm_client
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_iterations = max_iterations
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)
        print(f"Embedding model loaded: {embedding_model}")
        
        self.notebook: List[NotebookEntry] = []
        self.explored_nodes: Set[str] = set()
        self.current_iteration = 0
        
        self.entities = {}
        self.documents = {}
        # self.communities = {}
        self._index_nodes()
        self.query_embedding = None
        self.community = self._build_community_index
        
        # Add embeddings to nodes if not already present
        self._ensure_node_embeddings()

    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for given text"""
        embedding = self.embedder.encode(text)
        # Convert to numpy array if it's a tensor
        if hasattr(embedding, 'numpy'):
            return embedding.numpy()
        return np.array(embedding)
    
    def _ensure_node_embeddings(self):
        """Ensure all nodes have embeddings, generate if missing or inconsistent dimensions"""
        nodes_without_embeddings = []
        nodes_with_wrong_dimensions = []
        expected_dim = None
        
        # First pass: check what we have
        for node_id, node_data in self.graph.nodes(data=True):
            if 'embedding' not in node_data:
                nodes_without_embeddings.append(node_id)
            else:
                emb_array = np.array(node_data['embedding'])
                if expected_dim is None:
                    expected_dim = emb_array.shape[0]
                elif emb_array.shape[0] != expected_dim:
                    nodes_with_wrong_dimensions.append(node_id)
        
        # Get the dimension of our current embedding model
        test_embedding = self.embed("test")
        current_model_dim = test_embedding.shape[0]
        
        # Decide what to do based on the situation
        if not nodes_without_embeddings and not nodes_with_wrong_dimensions:
            if expected_dim == current_model_dim:
                print("All nodes already have embeddings with correct dimensions.")
                return
            else:
                print(f"All nodes have embeddings but wrong dimensions ({expected_dim} vs {current_model_dim})")
                print("Regenerating all embeddings with current model...")
                self._add_embeddings_to_nodes(list(self.graph.nodes()))
        elif nodes_with_wrong_dimensions:
            print(f"Found dimension mismatch. Expected: {expected_dim}, Current model: {current_model_dim}")
            print("Regenerating all embeddings for consistency...")
            self._add_embeddings_to_nodes(list(self.graph.nodes()))
        else:
            print(f"Generating embeddings for {len(nodes_without_embeddings)} nodes...")
            self._add_embeddings_to_nodes(nodes_without_embeddings)
    
    def _add_embeddings_to_nodes(self, node_ids: Optional[List[str]] = None):
        """Add embeddings to nodes in the knowledge graph"""
        if node_ids is None:
            node_ids = list(self.graph.nodes())
        
        batch_size = 32  # Process in batches for efficiency
        for i in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[i:i + batch_size]
            texts = []
            
            for node_id in batch_ids:
                node_data = self.graph.nodes[node_id]
                text_content = self._extract_node_text(node_data)
                texts.append(text_content)
            
            # Generate embeddings for the batch
            embeddings = self.embedder.encode(texts)
            
            # Store embeddings in node data
            for j, node_id in enumerate(batch_ids):
                self.graph.nodes[node_id]['embedding'] = embeddings[j].tolist()
            
            if len(node_ids) > batch_size:
                print(f"  Processed batch {i//batch_size + 1}/{(len(node_ids)-1)//batch_size + 1}")
    
    def _extract_node_text(self, node_data: Dict[str, Any]) -> str:
        """Extract text content from a node for embedding"""
        text_content = ""
        
        # Prioritize fields by importance
        for field in ['description', 'text', 'summary', 'full_content', 'title']:
            if field in node_data and node_data[field]:
                text_content += " " + str(node_data[field])
        
        # If no text content, use labels and other metadata
        if not text_content.strip():
            labels = node_data.get('labels', [])
            if labels:
                text_content = " ".join(labels)
            
            # Add any other string fields
            for key, value in node_data.items():
                if isinstance(value, str) and key not in ['embedding', 'id']:
                    text_content += " " + value
        
        return text_content.strip() or "empty_node"
    
    def _index_nodes(self):
        """Index nodes by their types for efficient retrieval"""
        for node_id, node_data in self.graph.nodes(data=True):
            labels = node_data.get('labels', [])
            
            if '__Entity__' in labels or 'Person' in labels:
                self.entities[node_id] = node_data
            elif 'Document' in labels:
                self.documents[node_id] = node_data
            # elif '__Community__' in labels:
            #     self.communities[node_id] = node_data

    def _build_community_index(self) -> Dict[str, List[str]]:
        community_map = {}
        for source, target, data in self.graph.edges(data=True):
            if data.get("type") == "IN_COMMUNITY":
                community_map.setdefault(target, []).append(source)
        return community_map
    
    def _create_prompt(self, system_message: str, user_message: str, schema: str) -> str:
        """Create a formatted prompt for the LLM"""
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": f"{system_message}\n\nYou MUST adhere to this schema:\n{schema}"},
                {"role": "user", "content": user_message},
            ],
            tokenize=False,
            add_bos=True,
            add_generation_prompt=True,
        )
    
    def reset_agent_state(self):
        """Reset the agent's state for a new question"""
        self.notebook = []
        self.explored_nodes = set()
        self.current_iteration = 0
        # self.query_embedding = None

    
    def create_query_plan(self, question: str) -> QueryPlan:
        """Create a strategic plan for answering the given question"""
        system_message = """You are an expert knowledge graph exploration agent. Create a systematic plan 
        for answering questions using iterative graph exploration. Focus on what information you need to find 
        and how to search for it effectively."""
        
        user_message = f"""
        Question: {question}
        
        Create a detailed exploration plan for this question. The knowledge graph contains:
        - Entities: Specific concepts, people, conditions, treatments, etc.
        - Documents: Research papers and larger text chunks  
        - Relationships: CAUSES, IS_ASSOCIATED_WITH, MENTIONS, IN_COMMUNITY
        
        Your plan should guide iterative exploration to gather comprehensive information.
        """
        
        schema = """
        class ReasoningStep(BaseModel):
            step: str = Field(..., description="A reasoning step in the planning process")
            required_info: List[str] = Field(..., description="Types of information needed for this step")

        class QueryPlan(BaseModel):
            reasoning_steps: List[ReasoningStep] = Field(..., description="Step-by-step plan to answer the query")
            key_concepts: List[str] = Field(..., description="Key concepts that need to be found in the knowledge graph")
            search_strategy: str = Field(..., description="Strategy for searching the knowledge graph")
            expected_answer_type: str = Field(..., description="What type of answer is expected (causal, descriptive, comparative, etc.)")
        """
        
        prompt = self._create_prompt(system_message, user_message, schema)
        
        original_schema = self.vllm_client.schema
        self.vllm_client.schema = QueryPlan
        
        result = self.vllm_client(prompt, sampling_params={
            "n": 1, "min_tokens": 100, "max_tokens": 800, "temperature": 0.1
        })
        
        self.vllm_client.schema = original_schema
        return result
    
    def find_initial_nodes(self, plan: QueryPlan, top_k: int = 10) -> List[str]:
        """Find initial nodes to start exploration based on the query plan using hybrid scoring"""
        keywords = plan.key_concepts.copy()
        for step in plan.reasoning_steps:
            keywords.extend(step.required_info)
        
        # Generate query embedding
        query_text = " ".join(plan.key_concepts + sum((step.required_info for step in plan.reasoning_steps), []))
        self.query_embedding = self.embed(query_text)
        
        relevant_nodes = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for node_id, node_data in self.graph.nodes(data=True):
            # Calculate keyword-based score
            keyword_score = self._calculate_keyword_score(node_data, keywords_lower)
            
            # Calculate semantic similarity score
            semantic_score = self._calculate_semantic_score(node_data, self.query_embedding)
            
            # Hybrid scoring: combine keyword and semantic scores
            # Weight semantic similarity higher for better conceptual matching
            total_score = keyword_score + (semantic_score * 3.0)
            
            if total_score > 0:
                relevant_nodes.append((node_id, total_score, keyword_score, semantic_score))
        
        # Sort by total score
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Debug: show scoring breakdown for top nodes
        print(f"Top {min(5, len(relevant_nodes))} nodes by hybrid score:")
        for i, (node_id, total, keyword, semantic) in enumerate(relevant_nodes[:5]):
            print(f"  {i+1}. {node_id}: total={total:.3f} (keyword={keyword:.3f}, semantic={semantic:.3f})")
        
        return [node_id for node_id, _, _, _ in relevant_nodes[:top_k]]
    
    def _calculate_keyword_score(self, node_data: Dict[str, Any], keywords_lower: List[str]) -> float:
        """Calculate keyword-based relevance score for a node"""
        score = 0.0
        searchable_text = ""
        
        # Extract searchable text from node
        for field in ['description', 'text', 'summary', 'full_content']:
            if field in node_data:
                searchable_text += " " + str(node_data[field])
        searchable_text = searchable_text.lower()
        
        # Count keyword occurrences
        for keyword in keywords_lower:
            if keyword in searchable_text:
                # Use log to prevent single keywords from dominating
                score += np.log(searchable_text.count(keyword) + 1)
        
        return score
    
    def _calculate_semantic_score(self, node_data: Dict[str, Any], query_embedding: np.ndarray) -> float:
        """Calculate semantic similarity score for a node"""
        if 'embedding' not in node_data:
            return 0.0
        
        try:
            node_embedding = np.array(node_data['embedding']).reshape(1, -1)
            query_emb = query_embedding.reshape(1, -1)
            similarity = cosine_similarity(query_emb, node_embedding)[0][0]
            
            # Convert similarity (-1 to 1) to positive score (0 to 1)
            return max(0.0, similarity)
        except Exception as e:
            print(f"Error calculating semantic score: {e}")
            return 0.0
    
    def extract_information_from_node(self, node_id: str, question: str, plan: QueryPlan) -> Optional[NotebookEntry]:
        """Extract relevant information from a specific node"""
        if node_id not in self.graph:
            return None
        
        node_data = self.graph.nodes[node_id]
        
        node_info = f"Node ID: {node_id}\n"
        node_info += f"Labels: {node_data.get('labels', [])}\n"
        
        if 'description' in node_data:
            node_info += f"Description: {node_data['description']}\n"
        if 'text' in node_data:
            node_info += f"Text: {node_data['text'][:1000]}{'...' if len(str(node_data['text'])) > 1000 else ''}\n"
        if 'summary' in node_data:
            node_info += f"Summary: {node_data['summary']}\n"
        if 'full_content' in node_data:
            node_info += f"Full Content: {str(node_data['full_content'])[:500]}{'...' if len(str(node_data.get('full_content', ''))) > 500 else ''}\n"
        
        
        neighbors = list(self.graph.neighbors(node_id))
        if neighbors:
            node_info += f"Connected to {len(neighbors)} other nodes\n"
        
        system_message = """You are an expert information extractor. Extract the most relevant and useful 
        information from the given node that helps answer the question. Focus on key facts, relationships, 
        and insights."""
        
        user_message = f"""
        Question: {question}
        Query Plan: {plan.model_dump_json(indent=2)}
        
        Node Information:
        {node_info}
        
        Extract the most relevant information from this node. Determine its relevance score and information type.
        """
        
        schema = """
        class NotebookEntry(BaseModel):
            source_node_id: str = Field(..., description="ID of the node this information came from")
            information: str = Field(..., description="Key information extracted from the node")
            relevance_score: float = Field(..., description="How relevant this information is (0-1)")
            information_type: str = Field(..., description="Type of information (causal, descriptive, statistical, etc.)")
        """
        
        prompt = self._create_prompt(system_message, user_message, schema)
        
        original_schema = self.vllm_client.schema
        self.vllm_client.schema = NotebookEntry
        
        try:
            result = self.vllm_client(prompt, sampling_params={
                "n": 1, "min_tokens": 50, "max_tokens": 400, "temperature": 0.1
            })
            self.vllm_client.schema = original_schema
            return result
        except Exception as e:
            print(f"Error extracting from node {node_id}: {e}")
            self.vllm_client.schema = original_schema
            return None
    
    def decide_next_exploration(self, question: str, plan: QueryPlan) -> ExplorationDecision:
        """Decide whether to continue exploring and what to explore next"""
        
        notebook_summary = "\n".join([
            f"- {entry.information} (relevance: {entry.relevance_score:.2f}, type: {entry.information_type})"
            for entry in self.notebook
        ])
        
        system_message = """You are an expert research agent. Based on the information gathered so far, 
        decide whether you have enough information to answer the question or if you need to explore more. 
        If exploring more, specify what nodes or areas to focus on next."""
        
        user_message = f"""
        Question: {question}
        Query Plan: {plan.model_dump_json(indent=2)}
        Current Iteration: {self.current_iteration + 1}/{self.max_iterations}
        
        Information Gathered So Far:
        {notebook_summary if notebook_summary else "No information gathered yet"}
        
        Explored Nodes: {list(self.explored_nodes)}
        
        Should you continue exploring? If yes, what should you explore next?
        """
        
        schema = """
        class ExplorationDecision(BaseModel):
            should_continue: bool = Field(..., description="Whether to continue exploring")
            reasoning: str = Field(..., description="Reasoning for the decision")
            next_nodes_to_explore: List[str] = Field(default=[], description="Specific node IDs to explore next")
            exploration_strategy: str = Field(..., description="How to explore next (neighbors, keywords, specific_nodes)")
            information_gaps: List[str] = Field(default=[], description="What information is still needed")
        """
        
        prompt = self._create_prompt(system_message, user_message, schema)
        
        original_schema = self.vllm_client.schema
        self.vllm_client.schema = ExplorationDecision
        
        result = self.vllm_client(prompt, sampling_params={
            "n": 1, "min_tokens": 100, "max_tokens": 600, "temperature": 0.2
        })
        
        self.vllm_client.schema = original_schema
        return result
    
    def get_neighbor_nodes(self, node_ids: List[str], max_neighbors: int = 15) -> List[str]:
        """Get neighboring nodes for further exploration with semantic ranking"""
        neighbors = set()
        
        for node_id in node_ids:
            if node_id in self.graph:
                node_neighbors = list(self.graph.neighbors(node_id))
                neighbors.update(node_neighbors)
        
        # Remove already explored nodes
        neighbors -= self.explored_nodes
        
        # If we have a query embedding, rank neighbors by semantic similarity
        if self.query_embedding is not None and len(neighbors) > max_neighbors:
            neighbor_scores = []
            
            for neighbor_id in neighbors:
                node_data = self.graph.nodes[neighbor_id]
                semantic_score = self._calculate_semantic_score(node_data, self.query_embedding)
                neighbor_scores.append((neighbor_id, semantic_score))
            
            # Sort by semantic score and take top neighbors
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in neighbor_scores[:max_neighbors]]
        
        return list(neighbors)[:max_neighbors]
    
    def find_semantically_similar_nodes(self, concepts: List[str], top_k: int = 10, exclude_explored: bool = True) -> List[str]:
        """Find nodes most semantically similar to given concepts"""
        if not concepts:
            return []
        
        # Create embedding for the concepts
        concept_text = " ".join(concepts)
        concept_embedding = self.embed(concept_text)
        
        similar_nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            # Skip explored nodes if requested
            if exclude_explored and node_id in self.explored_nodes:
                continue
            
            semantic_score = self._calculate_semantic_score(node_data, concept_embedding)
            
            if semantic_score > 0.1:  # Only consider nodes with reasonable similarity
                similar_nodes.append((node_id, semantic_score))
        
        # Sort by similarity and return top nodes
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in similar_nodes[:top_k]]
    
    def generate_final_answer(self, question: str, plan: QueryPlan) -> FinalAnswer:
        """Generate the final answer using all gathered information"""
        
        sorted_entries = sorted(self.notebook, key=lambda x: x.relevance_score, reverse=True)
        
        notebook_content = ""
        for i, entry in enumerate(sorted_entries, 1):
            notebook_content += f"{i}. Source: Node {entry.source_node_id}\n"
            notebook_content += f"   Information: {entry.information}\n"
            notebook_content += f"   Type: {entry.information_type}, Relevance: {entry.relevance_score:.2f}\n\n"
        
        system_message = """You are an expert researcher synthesizing information to provide a comprehensive answer. 
        Use all the gathered information from your notebook to construct a well-reasoned, complete response."""
        
        user_message = f"""
        Question: {question}
        Query Plan: {plan.model_dump_json(indent=2)}
        
        Information from Knowledge Graph Exploration:
        {notebook_content}
        
        Total iterations completed: {self.current_iteration}
        Total nodes explored: {len(self.explored_nodes)}
        
        Provide a comprehensive answer with clear reasoning steps, confidence assessment, and completeness evaluation.
        """
        
        schema = """
        class FinalAnswer(BaseModel):
            reasoning_steps: List[str] = Field(..., description="Step-by-step reasoning using gathered information")
            answer: str = Field(..., description="Final comprehensive answer to the question")
            confidence: float = Field(..., description="Confidence score (0-1) in the answer")
            sources: List[str] = Field(..., description="Node IDs used as sources for the answer")
            information_completeness: float = Field(..., description="How complete the gathered information is (0-1)")
        """
        
        prompt = self._create_prompt(system_message, user_message, schema)
        
        original_schema = self.vllm_client.schema
        self.vllm_client.schema = FinalAnswer
        
        result = self.vllm_client(prompt, sampling_params={
            "n": 1, "min_tokens": 300, "max_tokens": 2000, "temperature": 0.1
        })
        
        self.vllm_client.schema = original_schema
        return result
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Complete iterative pipeline to answer a question using the knowledge graph
        """
        print(f"Starting iterative exploration for: {question}")
        
        self.reset_agent_state()
        
        # Step 1: Create query plan
        print("Creating query plan...")
        plan = self.create_query_plan(question)
        print(f"Plan created with {len(plan.reasoning_steps)} steps")
        
        # Step 2: Find initial nodes
        print("  Finding initial nodes...")
        initial_nodes = self.find_initial_nodes(plan, top_k=8)
        print(f"   Found {len(initial_nodes)} initial nodes to explore")
        
        exploration_log = []
        
        # Step 3: Iterative exploration
        while self.current_iteration < self.max_iterations:
            print(f"\nIteration {self.current_iteration + 1}/{self.max_iterations}")
            
            # Determine nodes to explore this iteration
            if self.current_iteration == 0:
                nodes_to_explore = initial_nodes
            else:
                # Make exploration decision
                decision = self.decide_next_exploration(question, plan)
                exploration_log.append(decision)
                
                if not decision.should_continue:
                    print(f"Agent decided to stop: {decision.reasoning}")
                    break
                
                if decision.next_nodes_to_explore:
                    nodes_to_explore = decision.next_nodes_to_explore
                elif decision.exploration_strategy == "neighbors":
                    # Explore neighbors of high-relevance nodes
                    high_relevance_nodes = [entry.source_node_id for entry in self.notebook 
                                          if entry.relevance_score > 0.7]
                    nodes_to_explore = self.get_neighbor_nodes(high_relevance_nodes or [entry.source_node_id for entry in self.notebook[-3:]])
                else:
                    # Find new nodes based on information gaps using semantic similarity
                    gap_concepts = decision.information_gaps or plan.key_concepts
                    nodes_to_explore = self.find_semantically_similar_nodes(gap_concepts, top_k=5)
            
            nodes_to_explore = [n for n in nodes_to_explore if n not in self.explored_nodes]
            
            if not nodes_to_explore:
                print("No new nodes to explore")
                break
            
            print(f"Exploring {len(nodes_to_explore)} nodes...")
            
            # Extract information from nodes
            for node_id in nodes_to_explore[:5]:  # Limit to 5 nodes per iter
                if node_id not in self.explored_nodes:
                    entry = self.extract_information_from_node(node_id, question, plan)
                    if entry and entry.relevance_score > 0.3: 
                        self.notebook.append(entry)
                        print(f"     Added info from node {node_id} (relevance: {entry.relevance_score:.2f})")
                    
                    self.explored_nodes.add(node_id)
            
            self.current_iteration += 1
        
        print("\n Generating final answer...")
        final_answer = self.generate_final_answer(question, plan)
        
        print(f" Exploration Summary:")
        print(f"   - Total iterations: {self.current_iteration}")
        print(f"   - Nodes explored: {len(self.explored_nodes)}")
        print(f"   - Information gathered: {len(self.notebook)} entries")
        print(f"   - Final confidence: {final_answer.confidence:.2f}")
        
        return {
            "question": question,
            "plan": plan,
            "exploration_log": exploration_log,
            "notebook": self.notebook,
            "explored_nodes": list(self.explored_nodes),
            "iterations_completed": self.current_iteration,
            "final_answer": final_answer
        }
