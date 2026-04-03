import numpy as np


weights     = np.array([-0.02, 0.005, -0.3, 1.2, -0.1])
passenger_a = np.array([25, 100, 1, 1, 2])  # 1st class woman
passenger_b = np.array([35, 8,   3, 0, 1])  # 3rd class man

print(f"1st class woman score: {np.dot(weights, passenger_a):.2f}")
print(f"3rd class man score:   {np.dot(weights, passenger_b):.2f}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

king   = np.array([0.9, 0.1, 0.8])
queen  = np.array([0.8, 0.9, 0.7])
banana = np.array([0.1, 0.2, 0.9])

print(f"king vs queen:  {cosine_similarity(king, queen):.3f}")
print(f"king vs banana: {cosine_similarity(king, banana):.3f}")


# Semantic search using cosine similarity
# This is exactly how RAG retrieval works in Week 18
query     = np.array([0.8, 0.6, 0.2])
documents = np.array([[0.9, 0.5, 0.1],
                      [0.1, 0.2, 0.9],
                      [0.7, 0.7, 0.3]])

similarities = [cosine_similarity(query, doc) for doc in documents]
best_match   = np.argmax(similarities)
print(f"Similarities: {[round(s,3) for s in similarities]}")
print(f"Best matching document: {best_match}")