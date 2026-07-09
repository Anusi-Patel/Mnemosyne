from sentence_transformers import SentenceTransformer
import numpy as np

print("MNEMOSYNE: THE BRAIN UPGRADE")

# 1. Load a pre-trained open-source embedding model
# This model turns any text into a 384-dimensional vector
print("Waking up the Transformer model (this might take a few seconds to download)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Let's create two pieces of code and a query
code_chunk_1 = "def verify_password(plain_password, password_hash):\n    return hash(plain_password) == password_hash"
code_chunk_2 = "button { background-color: red; color: white; }"

query = "How is user authentication handled?"

# 3. Generate REAL Dense Vectors
# Notice how simple this is compared to writing your own logic?
vector_1 = model.encode(code_chunk_1)
vector_2 = model.encode(code_chunk_2)
query_vector = model.encode(query)

print(f"\nShape of our new vectors: {vector_1.shape}") # Should be (384,)

# 4. The math remains the same! (Dot Product)
# Because sentence-transformers normalizes output, dot product = cosine similarity
similarity_1 = np.dot(query_vector, vector_1)
similarity_2 = np.dot(query_vector, vector_2)

print("\n--- RESULTS ---")
print(f"Similarity to Auth Code: {similarity_1:.4f}")
print(f"Similarity to CSS Code:  {similarity_2:.4f}")

if similarity_1 > similarity_2:
    print("\n[Mnemosyne]: The Transformer correctly identified the authentication logic, even without exact keyword matches.")