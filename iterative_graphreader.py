"""Iterative graph reading agent extracted from graphreader.ipynb."""

import networkx as nx
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
    """Iteratively explores a GML knowledge graph to answer questions."""

    def __init__(self, gml_file_path: str, vllm_client, tokenizer_name: str = "mistralai/Mistral-7B-Instruct-v0.3", max_iterations: int = 5):
        self.graph = nx.read_gml(gml_file_path)
        self.vllm_client = vllm_client
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_iterations = max_iterations

        self.notebook: List[NotebookEntry] = []
        self.explored_nodes: Set[str] = set()
        self.current_iteration = 0

        self.entities = {}
        self.documents = {}
        self._index_nodes()
        self.query_embedding = None
        self.community = self._build_community_index

    def _index_nodes(self) -> None:
        for node_id, node_data in self.graph.nodes(data=True):
            labels = node_data.get('labels', [])
            if '__Entity__' in labels or 'Person' in labels:
                self.entities[node_id] = node_data
            elif 'Document' in labels:
                self.documents[node_id] = node_data

    def _build_community_index(self) -> Dict[str, List[str]]:
        community_map: Dict[str, List[str]] = {}
        for source, target, data in self.graph.edges(data=True):
            if data.get("type") == "IN_COMMUNITY":
                community_map.setdefault(target, []).append(source)
        return community_map

    def _create_prompt(self, system_message: str, user_message: str, schema: str) -> str:
        return self.tokenizer.apply_chat_template([
            {"role": "system", "content": f"{system_message}\n\nYou MUST adhere to this schema:\n{schema}"},
            {"role": "user", "content": user_message},
        ], tokenize=False, add_bos=True, add_generation_prompt=True)

    def reset_agent_state(self) -> None:
        self.notebook = []
        self.explored_nodes = set()
        self.current_iteration = 0

    def create_query_plan(self, question: str) -> QueryPlan:
        system_message = (
            "You are an expert knowledge graph exploration agent. Create a systematic plan"
            " for answering questions using iterative graph exploration. Focus on what information you need to find"
            " and how to search for it effectively."
        )
        user_message = (
            f"Question: {question}\n\n"
            "Create a detailed exploration plan for this question. The knowledge graph contains:\n"
            "- Entities: Specific concepts, people, conditions, treatments, etc.\n"
            "- Documents: Research papers and larger text chunks\n"
            "- Relationships: CAUSES, IS_ASSOCIATED_WITH, MENTIONS, IN_COMMUNITY\n\n"
            "Your plan should guide iterative exploration to gather comprehensive information."
        )
        schema = (
            "class ReasoningStep(BaseModel):\n"
            "    step: str = Field(..., description='A reasoning step in the planning process')\n"
            "    required_info: List[str] = Field(..., description='Types of information needed for this step')\n\n"
            "class QueryPlan(BaseModel):\n"
            "    reasoning_steps: List[ReasoningStep] = Field(..., description='Step-by-step plan to answer the query')\n"
            "    key_concepts: List[str] = Field(..., description='Key concepts that need to be found in the knowledge graph')\n"
            "    search_strategy: str = Field(..., description='Strategy for searching the knowledge graph')\n"
            "    expected_answer_type: str = Field(..., description='What type of answer is expected (causal, descriptive, comparative, etc.)')"
        )
        prompt = self._create_prompt(system_message, user_message, schema)
        original_schema = self.vllm_client.schema
        self.vllm_client.schema = QueryPlan
        result = self.vllm_client(prompt, sampling_params={"n": 1, "min_tokens": 100, "max_tokens": 800, "temperature": 0.1})
        self.vllm_client.schema = original_schema
        return result

    def find_initial_nodes(self, plan: QueryPlan, top_k: int = 10) -> List[str]:
        keywords = plan.key_concepts.copy()
        for step in plan.reasoning_steps:
            keywords.extend(step.required_info)
        relevant_nodes = []
        keywords_lower = [kw.lower() for kw in keywords]
        for node_id, node_data in self.graph.nodes(data=True):
            score = 0
            searchable_text = ""
            for field in ['description', 'text', 'summary', 'full_content']:
                if field in node_data:
                    searchable_text += " " + str(node_data[field])
            searchable_text = searchable_text.lower()
            for keyword in keywords_lower:
                if keyword in searchable_text:
                    score += searchable_text.count(keyword)
            if score > 0:
                relevant_nodes.append((node_id, score))
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in relevant_nodes[:top_k]]

    def extract_information_from_node(self, node_id: str, question: str, plan: QueryPlan) -> Optional[NotebookEntry]:
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
        system_message = (
            "You are an expert information extractor. Extract the most relevant and useful "
            "information from the given node that helps answer the question. Focus on key facts, relationships, "
            "and insights."
        )
        user_message = (
            f"Question: {question}\n"
            f"Query Plan: {plan.model_dump_json(indent=2)}\n\n"
            "Node Information:\n"
            f"{node_info}\n\n"
            "Extract the most relevant information from this node. Determine its relevance score and information type."
        )
        schema = (
            "class NotebookEntry(BaseModel):\n"
            "    source_node_id: str = Field(..., description='ID of the node this information came from')\n"
            "    information: str = Field(..., description='Key information extracted from the node')\n"
            "    relevance_score: float = Field(..., description='How relevant this information is (0-1)')\n"
            "    information_type: str = Field(..., description='Type of information (causal, descriptive, statistical, etc.)')"
        )
        prompt = self._create_prompt(system_message, user_message, schema)
        original_schema = self.vllm_client.schema
        self.vllm_client.schema = NotebookEntry
        try:
            result = self.vllm_client(prompt, sampling_params={"n": 1, "min_tokens": 50, "max_tokens": 400, "temperature": 0.1})
            self.vllm_client.schema = original_schema
            return result
        except Exception:
            self.vllm_client.schema = original_schema
            return None

    def decide_next_exploration(self, question: str, plan: QueryPlan) -> ExplorationDecision:
        notebook_summary = "\n".join(
            [f"- {entry.information} (relevance: {entry.relevance_score:.2f}, type: {entry.information_type})" for entry in self.notebook]
        )
        system_message = (
            "You are an expert research agent. Based on the information gathered so far, "
            "decide whether you have enough information to answer the question or if you need to explore more. "
            "If exploring more, specify what nodes or areas to focus on next."
        )
        user_message = (
            f"Question: {question}\n"
            f"Query Plan: {plan.model_dump_json(indent=2)}\n"
            f"Current Iteration: {self.current_iteration + 1}/{self.max_iterations}\n\n"
            f"Information Gathered So Far:\n{notebook_summary if notebook_summary else 'No information gathered yet'}\n\n"
            f"Explored Nodes: {list(self.explored_nodes)}\n\n"
            "Should you continue exploring? If yes, what should you explore next?"
        )
        schema = (
            "class ExplorationDecision(BaseModel):\n"
            "    should_continue: bool = Field(..., description='Whether to continue exploring')\n"
            "    reasoning: str = Field(..., description='Reasoning for the decision')\n"
            "    next_nodes_to_explore: List[str] = Field(default=[], description='Specific node IDs to explore next')\n"
            "    exploration_strategy: str = Field(..., description='How to explore next (neighbors, keywords, specific_nodes)')\n"
            "    information_gaps: List[str] = Field(default=[], description='What information is still needed')"
        )
        prompt = self._create_prompt(system_message, user_message, schema)
        original_schema = self.vllm_client.schema
        self.vllm_client.schema = ExplorationDecision
        result = self.vllm_client(prompt, sampling_params={"n": 1, "min_tokens": 100, "max_tokens": 600, "temperature": 0.2})
        self.vllm_client.schema = original_schema
        return result

    def get_neighbor_nodes(self, node_ids: List[str], max_neighbors: int = 15) -> List[str]:
        neighbors = set()
        for node_id in node_ids:
            if node_id in self.graph:
                node_neighbors = list(self.graph.neighbors(node_id))
                neighbors.update(node_neighbors)
        neighbors -= self.explored_nodes
        return list(neighbors)[:max_neighbors]

    def generate_final_answer(self, question: str, plan: QueryPlan) -> FinalAnswer:
        sorted_entries = sorted(self.notebook, key=lambda x: x.relevance_score, reverse=True)
        notebook_content = ""
        for i, entry in enumerate(sorted_entries, 1):
            notebook_content += f"{i}. Source: Node {entry.source_node_id}\n"
            notebook_content += f"   Information: {entry.information}\n"
            notebook_content += f"   Type: {entry.information_type}, Relevance: {entry.relevance_score:.2f}\n\n"
        system_message = (
            "You are an expert researcher synthesizing information to provide a comprehensive answer. "
            "Use all the gathered information from your notebook to construct a well-reasoned, complete response."
        )
        user_message = (
            f"Question: {question}\n"
            f"Query Plan: {plan.model_dump_json(indent=2)}\n\n"
            f"Information from Knowledge Graph Exploration:\n{notebook_content}\n"
            f"Total iterations completed: {self.current_iteration}\n"
            f"Total nodes explored: {len(self.explored_nodes)}\n\n"
            "Provide a comprehensive answer with clear reasoning steps, confidence assessment, and completeness evaluation."
        )
        schema = (
            "class FinalAnswer(BaseModel):\n"
            "    reasoning_steps: List[str] = Field(..., description='Step-by-step reasoning using gathered information')\n"
            "    answer: str = Field(..., description='Final comprehensive answer to the question')\n"
            "    confidence: float = Field(..., description='Confidence score (0-1) in the answer')\n"
            "    sources: List[str] = Field(..., description='Node IDs used as sources for the answer')\n"
            "    information_completeness: float = Field(..., description='How complete the gathered information is (0-1)')"
        )
        prompt = self._create_prompt(system_message, user_message, schema)
        original_schema = self.vllm_client.schema
        self.vllm_client.schema = FinalAnswer
        result = self.vllm_client(prompt, sampling_params={"n": 1, "min_tokens": 300, "max_tokens": 2000, "temperature": 0.1})
        self.vllm_client.schema = original_schema
        return result

    def answer_question(self, question: str) -> Dict[str, Any]:
        print(f"Starting iterative exploration for: {question}")
        self.reset_agent_state()
        print("Creating query plan...")
        plan = self.create_query_plan(question)
        print(f"Plan created with {len(plan.reasoning_steps)} steps")
        print("  Finding initial nodes...")
        initial_nodes = self.find_initial_nodes(plan, top_k=8)
        print(f"   Found {len(initial_nodes)} initial nodes to explore")
        exploration_log = []
        while self.current_iteration < self.max_iterations:
            print(f"\nIteration {self.current_iteration + 1}/{self.max_iterations}")
            if self.current_iteration == 0:
                nodes_to_explore = initial_nodes
            else:
                decision = self.decide_next_exploration(question, plan)
                exploration_log.append(decision)
                if not decision.should_continue:
                    print(f"Agent decided to stop: {decision.reasoning}")
                    break
                if decision.next_nodes_to_explore:
                    nodes_to_explore = decision.next_nodes_to_explore
                elif decision.exploration_strategy == "neighbors":
                    high_relevance_nodes = [entry.source_node_id for entry in self.notebook if entry.relevance_score > 0.7]
                    nodes_to_explore = self.get_neighbor_nodes(high_relevance_nodes or [entry.source_node_id for entry in self.notebook[-3:]])
                else:
                    nodes_to_explore = self.find_initial_nodes(plan, top_k=5)
            nodes_to_explore = [n for n in nodes_to_explore if n not in self.explored_nodes]
            if not nodes_to_explore:
                print("No new nodes to explore")
                break
            print(f"Exploring {len(nodes_to_explore)} nodes...")
            for node_id in nodes_to_explore[:5]:
                if node_id not in self.explored_nodes:
                    entry = self.extract_information_from_node(node_id, question, plan)
                    if entry and entry.relevance_score > 0.3:
                        self.notebook.append(entry)
                        print(f"     \u2713 Added info from node {node_id} (relevance: {entry.relevance_score:.2f})")
                    self.explored_nodes.add(node_id)
            self.current_iteration += 1
        print("\n Generating final answer...")
        final_answer = self.generate_final_answer(question, plan)
        print(" Exploration Summary:")
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
            "final_answer": final_answer,
        }
