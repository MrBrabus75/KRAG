# README.md

## Introduction

Ce projet utilise un modèle de langage large (LLM) en combinaison avec un graphe de connaissances (KG) et une recherche augmentée (RAG) pour améliorer les résultats des réponses générées. Le but est de permettre au modèle de langage d'utiliser des documents structurés avec des relations et des nœuds pour fournir des réponses plus précises et pertinentes.

## Knowledge Graph (KG)

Un graphe de connaissances (KG) est une représentation graphique de l'information où les entités (nœuds) sont connectées par des relations (arêtes). Par exemple, dans un KG sur les films, une entité pourrait être un acteur et une autre un film, avec une relation "a joué dans" reliant les deux.

### Exemple de KG

Brad Pitt --[a joué dans]--> Fight Club
Edward Norton --[a joué dans]--> Fight Club
David Fincher --[réalisé par]--> Fight Club

## Retrieval-Augmented Generation (RAG)

La recherche augmentée (RAG) combine la recherche d'informations avec la génération de réponses. Lorsqu'une question est posée, le système recherche d'abord des documents pertinents, puis utilise ces documents pour générer une réponse plus précise.

### Exemple de RAG

1. **Question**: "Qui a réalisé Fight Club?"
2. **Recherche**: Identification du document contenant l'information.
3. **Génération**: Utilisation du document pour générer la réponse "Fight Club a été réalisé par David Fincher."

## Fonctionnement combiné de RAG et KG

### Introduction
L'intégration de la Recherche Augmentée (RAG) et du Graphe de Connaissances (KG) permet d'améliorer la précision des réponses en utilisant des documents structurés et les relations entre les entités.

### Étapes

1. **Extraction et Ingestion**
    - **Document** : "Inception est un film réalisé par Christopher Nolan. Les acteurs principaux sont Leonardo DiCaprio, Joseph Gordon-Levitt et Ellen Page."
    - **KG** :
        ```
        Christopher Nolan --[a réalisé]--> Inception
        Leonardo DiCaprio --[a joué dans]--> Inception
        Joseph Gordon-Levitt --[a joué dans]--> Inception
        Ellen Page --[a joué dans]--> Inception
        ```

2. **Recherche et Récupération**
    - **Question** : "Qui a réalisé Inception ?"
    - **Recherche** : Embedding de la question, recherche dans FAISS, récupération du document pertinent.

3. **Génération de Réponses**
    - **Utilisation de KG et Documents** : Le LLM utilise les documents récupérés et les relations du KG pour générer la réponse.
    - **Réponse** : "Inception a été réalisé par Christopher Nolan."

## Explication des Scripts

### `requirements.txt`
Ce fichier contient la liste des bibliothèques Python nécessaires pour exécuter le projet. Les principales dépendances incluent `faiss-cpu` pour l'indexation de similarité, `numpy` pour les opérations numériques, `openai` pour l'API OpenAI, et `networkx` pour la gestion des graphes.

### `vectore.py`
Ce script initialise et sauvegarde un index FAISS pour la recherche de similarité vectorielle. Il configure un index basé sur la distance euclidienne entre les vecteurs.
```
def initialize_faiss_index(dimension=1536):
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, "faiss_index.index")
    print("FAISS index initialized and saved to 'faiss_index.index'.")
```

### `ingest.py`

Ce script ingère des documents, les divise en chunks, extrait les embeddings et construit des graphes de connaissances. Il stocke les embeddings dans un index FAISS et sauvegarde les documents pour une utilisation future.
```
def process_pdfs(directory="data"):
    embeddings, documents, knowledge_graphs = []
    # Extraction de texte, calcul des embeddings, extraction des entités et relations
    return embeddings, documents, knowledge_graphs

def ingest_embeddings_to_faiss(embeddings):
    index.add(embeddings)
    faiss.write_index(index, "faiss_index.index")
    print(f"Ingested {len(embeddings)} document chunks into FAISS index.")`
```

### `visu.py`

Ce script charge l'index FAISS et visualise les statistiques des vecteurs, telles que la distribution des distances et les projections PCA en 2D et 3D des embeddings.
```
def visualize_index_stats():
    index = load_faiss_index()
    # Extraction et visualisation des vecteurs
    plt.show()
```

### `llm.py`

Ce script interagit avec l'API OpenAI pour générer des réponses basées sur les documents récupérés par l'index FAISS. Il convertit la question en embedding, recherche les documents pertinents, et utilise ces documents pour générer une réponse.
```
def answer_query(query):
    query_embedding = get_embedding(query)
    indices = query_faiss_index(query_embedding)
    documents = retrieve_documents(indices)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Use the following documents to answer the query, sinon utilise tes propres connaissances :\n\n{documents}\n\nQuery: {query}\nAnswer:"}
        ],
        max_tokens=4096
    )
    return response.choices[0].message.content.strip()
```

Ce README donne une présentation générale du projet, une explication des concepts de graphe de connaissances et de recherche augmentée, ainsi qu'une description des différents scripts et de leur fonctionnement.
