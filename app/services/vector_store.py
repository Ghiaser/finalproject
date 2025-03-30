import faiss, os, pickle
import numpy as np
from typing import List, Tuple, Optional
from models import VectorEntry


class VectorStore:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index_path = f"data/faiss_index_{dimension}.index"
        self.entries_path = f"data/faiss_entries_{dimension}.bin"

        if not os.path.exists(self.index_path):
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self.index = faiss.IndexFlatIP(
                dimension
            )  # Inner product index (normalized vectors -> cosine similarity)
        else:
            self.index = faiss.read_index(self.index_path)

        if not os.path.exists(self.entries_path):
            self.entries: List[VectorEntry] = []
        else:
            with open(self.entries_path, "rb") as f:
                self.entries = pickle.load(f)

    def __del__(self):
        """Save the index on destruction."""
        try:
            if hasattr(self, "index") and hasattr(self, "index_path"):
                faiss.write_index(self.index, self.index_path)
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def add_entry(self, entry: VectorEntry):
        """Add a new vector entry to the store."""
        if len(self.entries) == 0:
            self.entries.append(entry)
            self.index.add(np.array([entry.embedding], dtype=np.float32))
        else:
            # Check if vector already exists to avoid duplicates
            if not any(
                np.array_equal(entry.embedding, e.embedding) for e in self.entries
            ):
                self.entries.append(entry)
                self.index.add(np.array([entry.embedding], dtype=np.float32))

        with open(self.entries_path, "wb") as f:
            pickle.dump(self.entries, f)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[VectorEntry, float]]:
        """Search for similar vectors and return entries with similarity scores."""
        if len(self.entries) == 0:
            return []

        # Ensure query vector is 2D
        query_vector = query_vector.reshape(1, -1)

        # Perform the search
        scores, indices = self.index.search(query_vector, min(top_k, len(self.entries)))

        # Return results with their scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 indicates no match found
                results.append((self.entries[idx], float(score)))

        return results

    def __len__(self):
        return len(self.entries)
