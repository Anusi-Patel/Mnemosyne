import numpy as np

print("=" * 60)
print("MNEMOSYNE - WEEK 1, DAY 5: Broadcasting")
print("NumPy's Secret Superpower")
print("=" * 60)

# Exercise 1: The Problem Broadcasting Solves
print("\n=== Exercise 1: The Tedious Way (Without Broadcasting) ===")
print("Adding 10 to every element in a matrix...\n")

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Original matrix:")
print(matrix)

# The MANUAL way (don't do this!)
print("\n❌ The tedious way (using loops):")
result_manual = np.zeros_like(matrix)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        result_manual[i, j] = matrix[i, j] + 10

print(result_manual)

# The BROADCASTING way (do this!)
print("\n✓ The broadcasting way (automatic):")
result_broadcast = matrix + 10
print(result_broadcast)

print("\nSame result, but broadcasting is:")
print("• Cleaner (1 line vs 4 lines)")
print("• Faster (NumPy optimized)")
print("• More readable")

# Exercise 2: Understanding Broadcasting Rules
print("\n" + "=" * 60)
print("=== Exercise 2: How Broadcasting Works ===")
print("NumPy automatically 'stretches' smaller arrays\n")

# Scalar + Array
array_1d = np.array([1, 2, 3, 4, 5])
scalar = 100

result = array_1d + scalar
print(f"Array:  {array_1d}")
print(f"Scalar: {scalar}")
print(f"Result: {result}")
print("^ The scalar was 'broadcast' to [100, 100, 100, 100, 100]\n")

# Vector + Matrix
matrix_2d = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
vector = np.array([10, 20, 30])

result = matrix_2d + vector
print("Matrix:")
print(matrix_2d)
print(f"\nVector: {vector}")
print("\nResult:")
print(result)
print("^ The vector was broadcast to each row!")

# Exercise 3: Broadcasting Shapes
print("\n" + "=" * 60)
print("=== Exercise 3: The Broadcasting Rules ===")
print("When can two arrays be broadcast together?\n")

print("Rule 1: Arrays are compatible if their dimensions match OR")
print("        one of them is 1 (or missing)\n")

# Example 1: Compatible shapes
a = np.ones((3, 4))     # Shape: (3, 4)
b = np.ones((4,))       # Shape: (4,)

print(f"Array a shape: {a.shape}  →  (3, 4)")
print(f"Array b shape: {b.shape}    →  (4,)")
print("Broadcasting b to match: (4,) → (1, 4) → (3, 4)")
print("✓ Compatible! Result shape: (3, 4)\n")

# Example 2: Another compatible case
c = np.ones((5, 1))     # Shape: (5, 1)
d = np.ones((1, 4))     # Shape: (1, 4)

print(f"Array c shape: {c.shape}  →  (5, 1)")
print(f"Array d shape: {d.shape}  →  (1, 4)")
print("Broadcasting: (5, 1) → (5, 4)")
print("              (1, 4) → (5, 4)")
print("✓ Compatible! Result shape: (5, 4)\n")

# Example 3: Incompatible shapes
print("Example of INCOMPATIBLE shapes:")
print("Array e: (3, 4)")
print("Array f: (3, 5)")
print("❌ Can't broadcast! Inner dimensions (4 and 5) don't match")

# Exercise 4: Common Broadcasting Patterns
print("\n" + "=" * 60)
print("=== Exercise 4: Common Broadcasting Patterns ===")
print("Patterns you'll use constantly in ML:\n")

# Pattern 1: Adding bias to neural network output
print("Pattern 1: Adding bias to each sample")
network_output = np.array([
    [0.1, 0.2, 0.3],  # Sample 1
    [0.4, 0.5, 0.6],  # Sample 2
    [0.7, 0.8, 0.9]   # Sample 3
])
bias = np.array([0.5, 0.3, 0.1])  # One bias per neuron

output_with_bias = network_output + bias
print(f"Network output (3 samples, 3 neurons):\n{network_output}")
print(f"\nBias (one per neuron): {bias}")
print(f"\nAfter adding bias:\n{output_with_bias}")
print("^ Each row got the same bias added!\n")

# Pattern 2: Normalizing data
print("Pattern 2: Normalizing by column (feature scaling)")
data = np.array([
    [1, 200, 3000],
    [2, 250, 3500],
    [3, 180, 2800]
])

mean_per_column = np.mean(data, axis=0)  # Mean of each column
std_per_column = np.std(data, axis=0)    # Std of each column

normalized = (data - mean_per_column) / std_per_column

print(f"Original data:\n{data}")
print(f"\nMean per column: {mean_per_column}")
print(f"Std per column:  {std_per_column}")
print(f"\nNormalized data:\n{normalized}")
print("^ This is 'standardization' - essential for ML!\n")

# Pattern 3: Element-wise operations with different shapes
print("Pattern 3: Distance from each point to each point")
points_a = np.array([[1], [2], [3]])      # Column vector (3, 1)
points_b = np.array([[10, 20, 30]])       # Row vector (1, 3)

distances = np.abs(points_a - points_b)   # Broadcasting magic!

print(f"Points A (column):\n{points_a}")
print(f"\nPoints B (row): {points_b}")
print(f"\nDistances (every A to every B):\n{distances}")
print(f"Shape: {distances.shape} = (3, 3)")

# Exercise 5: Broadcasting in Neural Networks
print("\n" + "=" * 60)
print("=== Exercise 5: Neural Network with Broadcasting ===")
print("How broadcasting makes neural networks elegant:\n")

# Mini batch of data
batch_size = 4
input_features = 5
hidden_neurons = 3

inputs = np.random.randn(batch_size, input_features)
weights = np.random.randn(input_features, hidden_neurons)
biases = np.random.randn(hidden_neurons)  # Just a 1D array!

print(f"Inputs shape:  {inputs.shape}  (4 samples, 5 features)")
print(f"Weights shape: {weights.shape}  (5 features → 3 neurons)")
print(f"Biases shape:  {biases.shape}   (3 biases)\n")

# Forward pass
hidden = np.dot(inputs, weights)  # Shape: (4, 3)
print(f"After matrix multiply: {hidden.shape}")

# Add biases - broadcasting happens here!
hidden = hidden + biases  # (4, 3) + (3,) → (4, 3)
print(f"After adding biases:   {hidden.shape}")

print("\nWhat happened:")
print("• Biases (3,) broadcast to (1, 3) then to (4, 3)")
print("• Same biases added to each of the 4 samples")
print("• No loops needed!")

# Exercise 6: Common Broadcasting Mistakes
print("\n" + "=" * 60)
print("=== Exercise 6: Common Broadcasting Errors ===")
print("Understanding what DOESN'T work:\n")

print("Example 1: Incompatible shapes")
print("a = (3, 4)  and  b = (3, 5)")
print("❌ Can't broadcast! Neither dimension is 1\n")

print("Example 2: Need to add dimension")
print("a = (5, 3)  and  b = (3,)")
print("✓ This works! b broadcasts to (1, 3) then (5, 3)")
print("But what if you want to broadcast to columns?\n")

column_matrix = np.array([[1, 2, 3],
                         [4, 5, 6]])
column_vector = np.array([10, 20])  # Shape: (2,)

print(f"Matrix:\n{column_matrix}")
print(f"Shape: {column_matrix.shape}")
print(f"\nVector: {column_vector}")
print(f"Shape: {column_vector.shape}")

# This won't work as expected!
# result = column_matrix + column_vector  # ERROR!

# Need to reshape vector to column
column_vector_reshaped = column_vector.reshape(2, 1)  # Shape: (2, 1)
result = column_matrix + column_vector_reshaped

print(f"\nVector reshaped to: {column_vector_reshaped.shape}")
print(f"Result:\n{result}")
print("✓ Now it broadcasts correctly to each column!")

# Exercise 7: Advanced Broadcasting
print("\n" + "=" * 60)
print("=== Exercise 7: Advanced Broadcasting Tricks ===")
print("Elegant solutions to complex problems:\n")

# Computing all pairwise distances
points = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

print(f"Points:\n{points}")
print(f"Shape: {points.shape}  (3 points, 2 dimensions)\n")

# Expand dimensions for broadcasting
points_expanded_1 = points[:, np.newaxis, :]  # Shape: (3, 1, 2)
points_expanded_2 = points[np.newaxis, :, :]  # Shape: (1, 3, 2)

# Subtract all points from all other points
differences = points_expanded_1 - points_expanded_2  # Shape: (3, 3, 2)

# Calculate distances
distances = np.sqrt(np.sum(differences**2, axis=2))

print("Pairwise distances:")
print(distances)
print("\n^ Distance matrix (3x3)")
print("distances[i, j] = distance from point i to point j")
print("Diagonal is 0 (distance from point to itself)")

# Exercise 8: Broadcasting for Mnemosyne
print("\n" + "=" * 60)
print("=== Exercise 8: Broadcasting in Mnemosyne ===")
print("Making code search efficient with broadcasting:\n")

# Batch of queries
queries = np.array([
    [0.9, 0.1, 0.8],  # "authentication"
    [0.1, 0.9, 0.2],  # "database"
    [0.5, 0.5, 0.5]   # "generic code"
])

# Code embeddings
codes = np.array([
    [0.9, 0.1, 0.8],  # auth.py
    [0.1, 0.9, 0.2],  # db.py
    [0.8, 0.2, 0.7],  # verify.py
    [0.2, 0.8, 0.3]   # query.py
])

print(f"Queries shape: {queries.shape}  (3 queries)")
print(f"Codes shape:   {codes.shape}    (4 code files)\n")

# Compute all similarities at once!
# queries: (3, 3)
# codes.T: (3, 4)
# Result: (3, 4) - similarity of each query to each code
similarities = np.dot(queries, codes.T)

print("Similarity matrix:")
print(similarities)
print(f"\nShape: {similarities.shape}")
print("similarities[i, j] = similarity of query i to code j")

# Apply sigmoid to all at once (broadcasting!)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

confidences = sigmoid(similarities)

print("\nConfidence scores:")
print(confidences)

print("\n🎉 CONGRATULATIONS! 🎉")
print("You just computed 12 search results (3 queries × 4 codes)")
print("in TWO lines of code, thanks to broadcasting!")
print("=" * 60)

print("\n💡 KEY TAKEAWAYS:")
print("1. Broadcasting = automatic array stretching")
print("2. Eliminates need for explicit loops")
print("3. Makes code cleaner and faster")
print("4. Essential for batch processing in ML")
print("5. Understanding shapes prevents 90% of errors")

print("\n🔮 FOR MNEMOSYNE:")
print("Broadcasting enables:")
print("• Processing multiple queries at once")
print("• Adding biases to entire batches")
print("• Normalizing embeddings efficiently")
print("• Computing pairwise similarities")
print("• Scaling to thousands of code files")
print("\nWithout broadcasting: slow loops")
print("With broadcasting: blazing fast matrix operations")
print("=" * 60)

print("\n🎊 WEEK 1 COMPLETE! 🎊")
print("You've learned:")
print("✓ Day 1: Vectors - representing data")
print("✓ Day 2: Matrices - organizing data")
print("✓ Day 3: Dot products - measuring similarity")
print("✓ Day 4: Sigmoid - making decisions")
print("✓ Day 5: Broadcasting - elegant operations")
print("\nYou now understand the FOUNDATION of all deep learning!")
print("Tomorrow: Put it all together in a weekend project!")
print("=" * 60)