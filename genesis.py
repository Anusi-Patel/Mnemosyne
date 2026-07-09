import numpy as np

# IMAGINE: These are not just numbers.
# These represent the "meaning" of two different sentences.
# In the real Mnemosyne, an AI will generate these numbers for you.
# For now, we manually create them to understand the math.

# Vector A: Represents the question "Where is the login?"
query_vector = np.array([1, 0.5, 0.8])

# Vector B: Represents a code file about "Authentication Logic"
code_file_vector_1 = np.array([0.9, 0.4, 0.9])

# Vector C: Represents a code file about "CSS Styling" (totally different)
code_file_vector_2 = np.array([-0.5, 0.1, 0.0])

# THE CORE MECHANIC: The Dot Product
# This calculates how "aligned" or "similar" two vectors are.
similarity_score_1 = np.dot(query_vector, code_file_vector_1)
similarity_score_2 = np.dot(query_vector, code_file_vector_2)

print(f"Similarity to Auth File: {similarity_score_1:.4f}")
print(f"Similarity to CSS File:  {similarity_score_2:.4f}")

# The "Threshold": If similarity is high, Mnemosyne retrieves this file.
if similarity_score_1 > 1.0:
    print("\n[Mnemosyne]: I found the relevant code context.")
else:
    print("\n[Mnemosyne]: No relevant code found.")