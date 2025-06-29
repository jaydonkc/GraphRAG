import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def generate_notebook(query: str, retmax: int) -> nbf.NotebookNode:
    nb = new_notebook()
    cells = []
    cells.append(new_markdown_cell(f"# Data Collection & Knowledge Graph\nThis notebook was auto-generated for **{query}**."))
    cells.append(new_code_cell(f"user_query = \"{query}\"\nretmax = {retmax}"))
    cells.append(new_markdown_cell("## Retrieve Data from PubMed"))
    cells.append(new_code_cell("""from Bio import Entrez

Entrez.email = 'your.email@example.com'
handle = Entrez.esearch(db='pubmed', term=user_query, retmax=retmax)
record = Entrez.read(handle)
pubmed_ids = record['IdList']
print(pubmed_ids[:5])
"""))
    cells.append(new_markdown_cell("## Load Ontologies"))
    cells.append(new_code_cell("""import pronto

go = pronto.Ontology('http://purl.obolibrary.org/obo/go.owl')
do = pronto.Ontology('http://purl.obolibrary.org/obo/doid.owl')
hpo = pronto.Ontology('http://purl.obolibrary.org/obo/hp.owl')
"""))
    cells.append(new_markdown_cell("## Optional Web Scraping"))
    cells.append(new_code_cell("""import requests
from bs4 import BeautifulSoup

url = 'https://example-biomedical-resource.com'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
"""))
    cells.append(new_markdown_cell("## NLP Preprocessing"))
    cells.append(new_code_cell("""import spacy
nlp = spacy.load('en_core_sci_sm')

processed_docs = [nlp(text) for text in []]  # Replace with your abstracts list
"""))
    cells.append(new_markdown_cell("## Build Knowledge Graph with Neo4j"))
    cells.append(new_code_cell("""from neo4j import GraphDatabase

driver = GraphDatabase.driver('neo4j://localhost:7687', auth=('neo4j', 'password'))
with driver.session() as session:
    session.run('CREATE (g:Gene {name: $name})', name='BRCA1')
"""))
    nb['cells'] = cells
    return nb

def main():
    query = input('Enter the biomedical subject or topic: ')
    depth = input('Number of results to fetch from PubMed (default 100): ')
    try:
        retmax = int(depth)
    except ValueError:
        retmax = 100
    nb = generate_notebook(query, retmax)
    fname = 'data_collection_and_kg.ipynb'
    with open(fname, 'w') as f:
        nbf.write(nb, f)
    print(f'Notebook written to {fname}')

if __name__ == '__main__':
    main()
