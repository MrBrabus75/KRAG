# vectoredb.py
import faiss
import numpy as np

def initialize_faiss_index(dimension=1536):
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, "faiss_index.index")
    print("FAISS index initialized and saved to 'faiss_index.index'.")

if __name__ == "__main__":
    initialize_faiss_index()
