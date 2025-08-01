# src/data_retriever.py
import requests
from Bio import Entrez
from typing import List, Dict

Entrez.email = "your_email@example.com"  # Replace with your email

def fetch_pubmed_abstracts(query: str, retmax: int = 5) -> str:
    with Entrez.esearch(db="pubmed", term=query, retmax=retmax) as handle:
        record = Entrez.read(handle)
        ids = record["IdList"] if isinstance(record, dict) and "IdList" in record else []
    if not ids:
        return ""
    with Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text") as fetch_handle:
        return fetch_handle.read()

def search_europe_pmc(query: str, page_size: int = 3) -> list[dict]:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": page_size
    }
    return requests.get(url, params=params).json()["resultList"]["result"]

def fetch_pmc_fulltext(pmcid: str) -> str:
    handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
    xml = handle.read()
    handle.close()
    # you’ll want to strip XML tags or parse with ElementTree
    return xml

def _fetch_preprints_from(server: str, days: int = 30) -> List[Dict]:
    """
    Pull the most recent `days` worth of preprints from bioRxiv or medRxiv.
    Returns the raw JSON 'collection' list.
    """
    from datetime import datetime, timedelta
    
    # Calculate date range
    today = datetime.now()
    start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    url = f"https://api.biorxiv.org/details/{server}/{start_date}/{end_date}/0"
    res = requests.get(url, timeout=15)
    data = res.json()
    return data.get("collection", [])

def fetch_preprint_abstracts(query: str, retmax: int = 5) -> List[Dict]:
    """
    Search bioRxiv + medRxiv for preprints whose title or abstract
    contains the query string (case‑insensitive). Returns up to retmax items.
    Each item: { 'server', 'title', 'abstract', 'doi', 'url', 'date' }
    """
    query_lc = query.lower()
    candidates = []

    for server in ("biorxiv", "medrxiv"):
        entries = _fetch_preprints_from(server, days=14)  # Reduced to 14 days for better focus on recent content
        for ent in entries:
            title = ent.get("title", "")
            abstract = ent.get("abstract", "")
            if query_lc in title.lower() or query_lc in abstract.lower():
                candidates.append({
                    "server": server,
                    "title": title,
                    "abstract": abstract,
                    "doi": ent.get("doi"),
                    "url": ent.get("doi_url") or ent.get("url"),
                    "date": ent.get("date")
                })
                if len(candidates) >= retmax:
                    break
        if len(candidates) >= retmax:
            break

    return candidates