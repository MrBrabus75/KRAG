# llm.py
from openai import OpenAI
import faiss
import numpy as np

# Initialisation du client OpenAI
client = OpenAI(api_key='VOTRE_API_OPENAI')

def query_faiss_index(query_embedding, top_k=5):
    index = faiss.read_index("faiss_index.index")
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    return I[0]

def retrieve_documents(indices):
    with open("documents.txt", "r") as f:
        documents = f.readlines()
    return [documents[i].strip() for i in indices]

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return np.array(client.embeddings.create(input=[text], model=model).data[0].embedding)

def answer_query(query):
    query_embedding = get_embedding(query)
    indices = query_faiss_index(query_embedding)
    documents = retrieve_documents(indices)
    
    # Utiliser les documents pour générer une réponse avec LLM
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Use the following documents to answer the query, sinon utilise tes propres conaissances :\n\n{documents}\n\nQuery: {query}\nAnswer:"}
        ],
        max_tokens=4096
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    query = input("Posez votre question : ")
    answer = answer_query(query)
    print(f"Réponse: {answer}")
