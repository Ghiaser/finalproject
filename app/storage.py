import faiss
import numpy as np

dimension = 512  # גודל הווקטור
index = faiss.IndexFlatL2(dimension)

def add_vector_to_index(vector, vector_id):
    index.add_with_ids(np.array([vector]), np.array([vector_id]))

def search_vector(query_vector, top_k):
    distances, indices = index.search(np.array([query_vector]), top_k)
    return indices, distances
