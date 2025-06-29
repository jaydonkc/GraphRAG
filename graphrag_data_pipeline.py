# Data Collection and Knowledge Graph Construction
# Simplified pipeline for the GraphRAG project

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from Bio import Entrez
from neo4j import GraphDatabase
import spacy
from spacy.tokens import Doc

try:
    import openai
except ImportError:  # Optional dependency
    openai = None  # type: ignore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PubMedResult:
    pmid: str
    title: str
    abstract: str


@dataclass
class Chunk:
    text: str
    doc_pmid: str
    index: int


@dataclass
class KeyElement:
    text: str
    label: str
    doc_pmid: str
    chunk_index: int


def fetch_pubmed(query: str, email: str, retmax: int = 20) -> List[PubMedResult]:
    """Fetch PubMed abstracts via Entrez."""
    Entrez.email = email
    logger.info("Searching PubMed for '%s'", query)
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    record = Entrez.read(handle)
    ids = record["IdList"]
    results: List[PubMedResult] = []
    for pmid in ids:
        fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        text = fetch.read()
        title, _, abstract = text.partition("\n")
        results.append(PubMedResult(pmid=pmid, title=title.strip(), abstract=abstract.strip()))
    logger.info("Retrieved %d PubMed documents", len(results))
    return results


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks of approximately `size` tokens."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def extract_key_elements(nlp, text: str) -> Iterable[Tuple[str, str]]:
    """Extract named entities using spaCy."""
    doc: Doc = nlp(text)
    for ent in doc.ents:
        yield ent.text, ent.label_


def connect_neo4j(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))


def load_to_graph(driver, docs: List[PubMedResult], nlp) -> None:
    logger.info("Loading documents into Neo4j")
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS ON (d:Document) ASSERT d.pmid IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS ON (c:Chunk) ASSERT c.id IS UNIQUE")
        for doc in docs:
            session.run("MERGE (d:Document {pmid: $pmid, title: $title})", pmid=doc.pmid, title=doc.title)
            chunks = chunk_text(doc.abstract)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.pmid}_{i}"
                session.run(
                    "MERGE (c:Chunk {id: $id, text: $text})",
                    id=chunk_id,
                    text=chunk,
                )
                session.run(
                    "MATCH (d:Document {pmid: $pmid}), (c:Chunk {id: $cid})\nMERGE (d)-[:HAS_CHUNK]->(c)",
                    pmid=doc.pmid,
                    cid=chunk_id,
                )
                for ent_text, ent_label in extract_key_elements(nlp, chunk):
                    session.run(
                        "MERGE (k:KeyElement {text: $text, label: $label})",
                        text=ent_text,
                        label=ent_label,
                    )
                    session.run(
                        "MATCH (k:KeyElement {text: $text, label: $label}), (c:Chunk {id: $cid})\nMERGE (c)-[:HAS_KEY_ELEMENT]->(k)",
                        text=ent_text,
                        label=ent_label,
                        cid=chunk_id,
                    )


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Data Pipeline")
    parser.add_argument("query", help="Biomedical query")
    parser.add_argument("email", help="Email for Entrez")
    parser.add_argument("--retmax", type=int, default=10)
    parser.add_argument("--neo4j", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    docs = fetch_pubmed(args.query, args.email, args.retmax)
    driver = connect_neo4j(args.neo4j, args.user, args.password)
    load_to_graph(driver, docs, nlp)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
