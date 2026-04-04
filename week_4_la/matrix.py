import numpy as np

# create a matrix

x = np.array([[28,  7.25, 3],   
    [38, 71.28, 1],  
    [26,  7.92, 3],  
    [35, 53.10, 1]])

print("Matrix x:\n", x)
print("Shape of x:", x.shape)
print("row 1", x[0])
print("column 1", x[:,0])


weights = np.array([-0.02, 0.005, -0.3,1.2, -0.1])

rose = np.array([17, 71.28, 1, 1, 2])
jack = np.array([20, 7.25,  3, 0, 1])

# Dot product — the core prediction operation
rose_score = np.dot(weights, rose)
jack_score = np.dot(weights, jack)

print(f"Rose score: {rose_score:.3f}")
print(f"Jack score: {jack_score:.3f}")


# @ is cleaner for longer expressions

rose_score2 = weights @ rose
print(f"Rose score (using @): {rose_score2:.3f}")

# Verify by doing it manually — multiply then sum
manual = (weights * rose).sum()
print(f"Rose score (manual): {manual:.3f}")


# dot product for similarity
# used in cosine similarity and recommendation systems

def cosine_similarity(a,b):
    """How similar are two vectors? Returns
      a value between -1(opposite) and 1(identical)."""
    
    return np.dot(a,b)/ (np.linalg.norm(a) * np.linalg.norm(b))

# Simple word embeddings (normally 768 numbers — we use 3 for illustration)
word_king   = np.array([0.9, 0.1, 0.8])
word_queen  = np.array([0.8, 0.9, 0.7])
word_banana = np.array([0.1, 0.9, 0.1])

print(f"king vs queen:  {cosine_similarity(word_king, word_queen):.3f}")  
print(f"king vs banana: {cosine_similarity(word_king, word_banana):.3f}")


w = np.array([[-0.3],[0.5],[-0.1]])

print("Shape of w:", w.shape)
print('Shape of x:', x.shape)


predictions = x @ w
print("Predictions shape:", predictions.shape)



