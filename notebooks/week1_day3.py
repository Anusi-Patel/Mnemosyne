import numpy as np

print("=" * 60)
print("MNEMOSYNE - WEEK 1, DAY 3: The Dot Product")
print("The Engine of All Intelligence")
print("=" * 60)

# Exercise 1: Understanding the Dot Product Visually
print("\n=== Exercise 1: What IS a Dot Product? ===")
print("The dot product measures HOW SIMILAR two vectors are\n")

# Two vectors pointing in the same direction
same_direction_a = np.array([1, 0])
same_direction_b = np.array([2, 0])

dot_same = np.dot(same_direction_a, same_direction_b)
print(f"Vector A: {same_direction_a} (pointing right)")
print(f"Vector B: {same_direction_b} (pointing right)")
print(f"Dot product: {dot_same}")
print("^ HIGH value = vectors point in SAME direction\n")

# Two vectors pointing in opposite directions
opposite_a = np.array([1, 0])
opposite_b = np.array([-1, 0])

dot_opposite = np.dot(opposite_a, opposite_b)
print(f"Vector A: {opposite_a} (pointing right)")
print(f"Vector B: {opposite_b} (pointing LEFT)")
print(f"Dot product: {dot_opposite}")
print("^ NEGATIVE value = vectors point in OPPOSITE directions\n")

# Two perpendicular vectors
perpendicular_a = np.array([1, 0])
perpendicular_b = np.array([0, 1])

dot_perpendicular = np.dot(perpendicular_a, perpendicular_b)
print(f"Vector A: {perpendicular_a} (pointing right)")
print(f"Vector B: {perpendicular_b} (pointing up)")
print(f"Dot product: {dot_perpendicular}")
print("^ ZERO = vectors are PERPENDICULAR (completely different)\n")

print("🔑 KEY INSIGHT: Dot product measures SIMILARITY")
print("   High positive = similar")
print("   Zero = unrelated")
print("   Negative = opposite")

# Exercise 2: Computing Dot Products (The Math)
print("\n" + "=" * 60)
print("=== Exercise 2: How to Calculate Dot Product ===")
print("Two ways: by hand and with NumPy\n")

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}\n")

# Method 1: Manual calculation
manual = (1 * 4) + (2 * 5) + (3 * 6)
print(f"Manual calculation: (1×4) + (2×5) + (3×6) = {manual}")

# Method 2: Element-wise multiply then sum
element_wise = v1 * v2  # [1*4, 2*5, 3*6] = [4, 10, 18]
sum_result = np.sum(element_wise)  # 4 + 10 + 18 = 32
print(f"Element-wise: {v1} * {v2} = {element_wise}")
print(f"Then sum: {element_wise} -> {sum_result}")

# Method 3: NumPy's dot function
numpy_dot = np.dot(v1, v2)
print(f"NumPy dot: {numpy_dot}")

print(f"\nAll three methods give: {manual} = {sum_result} = {numpy_dot}")
print("✓ They're all the same! Use np.dot() - it's fastest.")

# Exercise 3: Dot Product for Similarity
print("\n" + "=" * 60)
print("=== Exercise 3: Measuring Similarity ===")
print("THIS is how AI finds 'related' things!\n")

# Imagine these are embeddings of different words
# (In reality, embeddings have 100s or 1000s of dimensions)

word_king = np.array([0.8, 0.6, 0.1, 0.9])
word_queen = np.array([0.7, 0.5, 0.2, 0.85])
word_car = np.array([0.1, 0.2, 0.9, 0.15])
word_vehicle = np.array([0.15, 0.25, 0.85, 0.2])

print("Word embeddings (simplified):")
print(f"'king':    {word_king}")
print(f"'queen':   {word_queen}")
print(f"'car':     {word_car}")
print(f"'vehicle': {word_vehicle}\n")

# Compare similarities
sim_king_queen = np.dot(word_king, word_queen)
sim_king_car = np.dot(word_king, word_car)
sim_car_vehicle = np.dot(word_car, word_vehicle)

print("Similarity scores (dot products):")
print(f"king ↔ queen:   {sim_king_queen:.3f}")
print(f"king ↔ car:     {sim_king_car:.3f}")
print(f"car ↔ vehicle:  {sim_car_vehicle:.3f}\n")

print("Notice:")
print(f"✓ 'king' and 'queen' are similar ({sim_king_queen:.3f})")
print(f"✓ 'car' and 'vehicle' are similar ({sim_car_vehicle:.3f})")
print(f"✓ 'king' and 'car' are NOT similar ({sim_king_car:.3f})")
print("\nThis is EXACTLY how search engines and recommendation work!")

# Exercise 4: Matrix-Vector Dot Product
print("\n" + "=" * 60)
print("=== Exercise 4: Matrix × Vector ===")
print("How neural networks process data\n")

# A weight matrix (3x4) - could be a neural layer
weights = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 0.8, 0.7, 0.6]
])

# An input vector (4 features)
input_vec = np.array([1.0, 2.0, 3.0, 4.0])

print(f"Weight matrix (3x4):\n{weights}\n")
print(f"Input vector (4): {input_vec}\n")

# Each row of weights takes dot product with input
output = np.dot(weights, input_vec)

print(f"Output (3): {output}\n")

print("What happened:")
print(f"Row 1 · input = {np.dot(weights[0], input_vec):.1f}")
print(f"Row 2 · input = {np.dot(weights[1], input_vec):.1f}")
print(f"Row 3 · input = {np.dot(weights[2], input_vec):.1f}")
print("\n^ Each neuron computes ONE dot product!")
print("A neural layer with 1000 neurons = 1000 dot products")

# Exercise 5: Matrix-Matrix Multiplication
print("\n" + "=" * 60)
print("=== Exercise 5: Matrix × Matrix ===")
print("Multiple inputs through a neural layer at once\n")

# 2 samples, 3 features each
inputs = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Transform 3 features into 2 outputs
weights_layer = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])

print(f"Inputs (2 samples × 3 features):\n{inputs}\n")
print(f"Weights (3 features → 2 outputs):\n{weights_layer}\n")

result = np.dot(inputs, weights_layer)

print(f"Result (2 samples × 2 outputs):\n{result}\n")

print("Shape rule for matrix multiplication:")
print(f"({inputs.shape[0]}, {inputs.shape[1]}) × ({weights_layer.shape[0]}, {weights_layer.shape[1]}) = ({result.shape[0]}, {result.shape[1]})")
print("(m, n) × (n, p) = (m, p)")
print("      ↑   ↑")
print("      Must match!")

# Exercise 6: Building Mnemosyne's Search
print("\n" + "=" * 60)
print("=== Exercise 6: Code Search with Dot Products ===")
print("THIS IS HOW MNEMOSYNE WILL WORK!\n")

# Imagine these are embeddings of code snippets
code_snippets = {
    "auth_login": np.array([0.9, 0.1, 0.2, 0.8]),
    "auth_logout": np.array([0.85, 0.15, 0.25, 0.75]),
    "database_query": np.array([0.1, 0.9, 0.8, 0.2]),
    "database_insert": np.array([0.15, 0.85, 0.75, 0.25]),
    "ui_button": np.array([0.2, 0.2, 0.1, 0.1])
}

# User's question (also embedded)
query = np.array([0.88, 0.12, 0.2, 0.82])
print(f"User asks: 'Where is the authentication code?'")
print(f"Query embedding: {query}\n")

print("Computing similarity to each code snippet:")
similarities = {}
for name, embedding in code_snippets.items():
    similarity = np.dot(query, embedding)
    similarities[name] = similarity
    print(f"{name:20s}: {similarity:.3f}")

# Find the most similar
best_match = max(similarities, key=similarities.get)
print(f"\n✨ Best match: {best_match}")
print(f"Similarity score: {similarities[best_match]:.3f}")

print("\n🎉 CONGRATULATIONS! 🎉")
print("You just implemented the CORE of Mnemosyne!")
print("This is semantic search - the foundation of RAG systems.")
print("=" * 60)

print("\n💡 CRITICAL INSIGHTS FOR TODAY:")
print("1. Dot product = measure of similarity")
print("2. Neural networks = chains of dot products")
print("3. Search engines = finding highest dot product")
print("4. Mnemosyne will use this to find relevant code")

print("\n🔮 FOR MNEMOSYNE:")
print("When you ask 'Where is the authentication logic?':")
print("1. Your question becomes a vector (embedding)")
print("2. Every code chunk is also a vector")
print("3. Compute dot product with each chunk")
print("4. Return chunks with highest dot products")
print("5. Send those to LLM for final answer")
print("\nYou now understand Mnemosyne's brain!")
print("=" * 60)