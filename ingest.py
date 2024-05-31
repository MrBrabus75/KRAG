# ingest.py
import os
from openai import OpenAI
import faiss
import numpy as np
import pdfplumber
import networkx as nx
import matplotlib.pyplot as plt

# Initialisation du client OpenAI
client = OpenAI(api_key='sk-proj-7RdMpFd7EzY56pDWEqueT3BlbkFJ677UrTKdVC8nZApPAONi')

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return np.array(client.embeddings.create(input=[text], model=model).data[0].embedding)

def chunk_text(text, max_tokens=4096):  # Pour ne pas dépasser la limite de tokens
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 pour l'espace ou la séparation
        if current_length + word_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def extract_entities_and_relations(text):
    prompt = f"Extract entities and their relationships from the following text:\n\n{text}\n\nOutput the result in the format 'Entity1 -[relation]-> Entity2'."
    messages = [
        {"role": "system", "content": "You are an information extraction assistant."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1024
    )
    result = response.choices[0].message.content.strip()  # Utilisation correcte de 'message.content'
    entities = set()
    relations = []
    for line in result.split('\n'):
        if '-[' in line and ']->' in line:
            parts = line.split('-[')
            entity1 = parts[0].strip()
            relation, entity2 = parts[1].split(']->')
            entities.add(entity1)
            entities.add(entity2.strip())
            relations.append((entity1, relation.strip(), entity2.strip()))
    return list(entities), relations

def build_knowledge_graph(entities, relations):
    G = nx.DiGraph()
    for entity in entities:
        G.add_node(entity)
    for subject, relation, object_ in relations:
        G.add_edge(subject, object_, label=relation)
    return G

def visualize_knowledge_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def process_pdfs(directory="data"):
    embeddings = []
    documents = []
    knowledge_graphs = []

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(directory, filename)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text() is not None)
                text_chunks = chunk_text(text)
                for chunk in text_chunks:
                    embedding = get_embedding(chunk)
                    embeddings.append(embedding)
                    documents.append(chunk)

                    entities, relations = extract_entities_and_relations(chunk)
                    G = build_knowledge_graph(entities, relations)
                    knowledge_graphs.append(G)

    return embeddings, documents, knowledge_graphs

def ingest_embeddings_to_faiss(embeddings):
    if not os.path.exists("faiss_index.index"):
        index = faiss.IndexFlatL2(1536)  # Assuming dimension is 1536
        faiss.write_index(index, "faiss_index.index")

    index = faiss.read_index("faiss_index.index")
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)
    faiss.write_index(index, "faiss_index.index")
    print(f"Ingested {len(embeddings)} document chunks into FAISS index.")

if __name__ == "__main__":
    embeddings, documents, knowledge_graphs = process_pdfs()
    ingest_embeddings_to_faiss(embeddings)
    # Sauvegarder les documents pour récupération future
    with open("documents.txt", "w") as f:
        for doc in documents:
            f.write("%s\n" % doc)

    # Visualiser le premier Knowledge Graph pour vérification
    if knowledge_graphs:
        visualize_knowledge_graph(knowledge_graphs[0])
