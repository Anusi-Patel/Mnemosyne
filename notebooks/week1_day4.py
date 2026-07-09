import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("MNEMOSYNE - WEEK 1, DAY 4: The Sigmoid Function")
print("From Numbers to Decisions")
print("=" * 60)

# Exercise 1: Understanding the Problem
print("\n=== Exercise 1: Why Do We Need Sigmoid? ===")
print("The problem with raw dot products:\n")

# Similarity scores from a search
similarities = {
    "auth.py": 15.7,
    "login.py": 12.3,
    "database.py": 2.1,
    "ui.py": -3.4,
    "tests.py": 0.8
}

print("Raw dot product scores:")
for file, score in similarities.items():
    print(f"{file:15s}: {score:6.1f}")

print("\nProblems:")
print("❌ Numbers can be ANY value (negative, huge, tiny)")
print("❌ Hard to interpret: Is 15.7 'good'? Is 2.1 'relevant'?")
print("❌ Can't represent as probability/confidence")
print("\n✓ Solution: Squish everything between 0 and 1")
print("✓ 0 = definitely not relevant")
print("✓ 1 = definitely relevant")
print("✓ 0.5 = uncertain")

# Exercise 2: The Sigmoid Formula
print("\n" + "=" * 60)
print("=== Exercise 2: The Magic Formula ===")
print("Sigmoid(x) = 1 / (1 + e^(-x))\n")

def sigmoid(x):
    """
    The sigmoid activation function
    Squishes any number to be between 0 and 1
    """
    return 1 / (1 + np.exp(-x))

# Test with different inputs
test_values = [-10, -5, -1, 0, 1, 5, 10]

print("What sigmoid does to numbers:")
print(f"{'Input':>8s} → {'Sigmoid(x)':>12s}")
print("-" * 25)
for x in test_values:
    result = sigmoid(x)
    print(f"{x:8.1f} → {result:12.6f}")

print("\n🔑 KEY INSIGHTS:")
print("• Very negative numbers → close to 0")
print("• Zero → exactly 0.5 (neutral)")
print("• Very positive numbers → close to 1")
print("• The transition is SMOOTH (no sudden jumps)")

# Exercise 3: Visualizing Sigmoid
print("\n" + "=" * 60)
print("=== Exercise 3: The Sigmoid Curve ===")
print("Creating a visualization...\n")

# Create many x values
x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)

# Print some key points
print("Key points on the curve:")
print(f"sigmoid(-5) = {sigmoid(-5):.6f}  (almost 0)")
print(f"sigmoid(0)  = {sigmoid(0):.6f}   (exactly 0.5)")
print(f"sigmoid(5)  = {sigmoid(5):.6f}   (almost 1)")

print("\nThe curve shape:")
print("      1.0 |           ___________")
print("          |         /")
print("      0.5 |       /")
print("          |     /")
print("      0.0 |___/")
print("          -5    0    5")
print("\nIt's an S-shaped curve!")

# Exercise 4: Applying Sigmoid to Our Search Results
print("\n" + "=" * 60)
print("=== Exercise 4: Converting Scores to Probabilities ===")
print("Making those raw scores interpretable:\n")

print("BEFORE sigmoid (raw dot products):")
for file, score in similarities.items():
    print(f"{file:15s}: {score:6.1f}")

print("\nAFTER sigmoid (probabilities):")
for file, score in similarities.items():
    probability = sigmoid(score)
    confidence = "HIGH" if probability > 0.8 else "MEDIUM" if probability > 0.5 else "LOW"
    print(f"{file:15s}: {probability:.4f} ({confidence})")

print("\n✨ Now we can say:")
print("• auth.py: 99.9999% confident it's relevant")
print("• database.py: 89% confident it's relevant")
print("• ui.py: Only 3% confident (probably not relevant)")

# Exercise 5: Sigmoid in Neural Networks
print("\n" + "=" * 60)
print("=== Exercise 5: Neurons Making Decisions ===")
print("How a single neuron uses sigmoid:\n")

# A simple neuron
inputs = np.array([0.5, 0.8, 0.3])
weights = np.array([0.6, 0.4, 0.2])
bias = 0.1

print(f"Inputs:  {inputs}")
print(f"Weights: {weights}")
print(f"Bias:    {bias}\n")

# Step 1: Weighted sum (dot product + bias)
weighted_sum = np.dot(inputs, weights) + bias
print(f"Step 1 - Weighted sum: {weighted_sum:.4f}")
print(f"         (dot product + bias)")

# Step 2: Apply sigmoid
activation = sigmoid(weighted_sum)
print(f"\nStep 2 - Apply sigmoid: {activation:.4f}")
print(f"         This is the neuron's output!")

if activation > 0.5:
    print(f"\n✓ Neuron fires! ({activation:.1%} confident)")
else:
    print(f"\n✗ Neuron doesn't fire ({activation:.1%} confident)")

print("\nThis is EXACTLY what happens in every neuron:")
print("1. Compute weighted sum of inputs")
print("2. Add bias")
print("3. Apply sigmoid")
print("4. Output a probability")

# Exercise 6: Binary Classification
print("\n" + "=" * 60)
print("=== Exercise 6: Making Binary Decisions ===")
print("Sigmoid for yes/no questions:\n")

# Examples: Is this email spam?
emails = {
    "Get rich quick!!!": np.array([0.9, 0.8, 0.95]),  # Spam indicators
    "Meeting tomorrow at 3pm": np.array([0.1, 0.2, 0.15]),  # Not spam
    "You won a prize!": np.array([0.85, 0.75, 0.9]),  # Probably spam
    "Here's the report": np.array([0.2, 0.15, 0.1])  # Not spam
}

# Simple spam detection weights
spam_weights = np.array([1.0, 0.8, 1.2])
spam_bias = -2.0  # Bias toward "not spam"

print("Email Spam Detector:")
print("-" * 50)

for email, features in emails.items():
    score = np.dot(features, spam_weights) + spam_bias
    probability = sigmoid(score)
    
    is_spam = "🚨 SPAM" if probability > 0.5 else "✓ Safe"
    print(f"{email:30s}: {probability:.1%} {is_spam}")

print("\nThe sigmoid threshold (0.5) determines the decision!")

# Exercise 7: Derivative of Sigmoid (Preview for Tomorrow)
print("\n" + "=" * 60)
print("=== Exercise 7: Why Sigmoid is Special ===")
print("Sigmoid has a beautiful mathematical property:\n")

def sigmoid_derivative(x):
    """The derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)

x_test = 0
sig = sigmoid(x_test)
deriv = sigmoid_derivative(x_test)

print(f"At x = {x_test}:")
print(f"  sigmoid({x_test}) = {sig:.4f}")
print(f"  derivative = {deriv:.4f}")

print("\nWhy this matters:")
print("• The derivative tells us how to adjust weights")
print("• This is how neural networks LEARN")
print("• Next week, you'll use this for backpropagation!")

# Exercise 8: Building Mnemosyne's Confidence Scorer
print("\n" + "=" * 60)
print("=== Exercise 8: Mnemosyne's Relevance Scorer ===")
print("Using sigmoid to rank search results:\n")

# Code chunks from a repository
code_chunks = {
    "def login(username, password):": np.array([0.9, 0.1, 0.8]),
    "def logout():": np.array([0.85, 0.15, 0.75]),
    "def query_database(sql):": np.array([0.1, 0.9, 0.2]),
    "class Button:": np.array([0.2, 0.1, 0.3]),
    "def authenticate_user():": np.array([0.95, 0.05, 0.9])
}

# User query: "authentication code"
query_embedding = np.array([0.9, 0.1, 0.85])

print("User query: 'Show me authentication code'")
print(f"Query embedding: {query_embedding}\n")

results = []
for code, embedding in code_chunks.items():
    # Calculate similarity
    raw_score = np.dot(query_embedding, embedding)
    
    # Convert to confidence with sigmoid
    confidence = sigmoid(raw_score)
    
    results.append((code, raw_score, confidence))

# Sort by confidence
results.sort(key=lambda x: x[2], reverse=True)

print("Ranked results:")
print("-" * 60)
for i, (code, raw, conf) in enumerate(results, 1):
    stars = "⭐" * int(conf * 5)  # Visual confidence indicator
    print(f"{i}. {code[:40]:40s}")
    print(f"   Raw score: {raw:6.3f} → Confidence: {conf:.1%} {stars}")
    print()

print("🎉 CONGRATULATIONS! 🎉")
print("You just built Mnemosyne's relevance scoring system!")
print("=" * 60)

print("\n💡 KEY TAKEAWAYS:")
print("1. Sigmoid squishes numbers to [0, 1]")
print("2. Turns any score into a probability/confidence")
print("3. Every neuron uses sigmoid (or similar function)")
print("4. Mnemosyne uses sigmoid to rank search results")
print("5. The S-curve shape makes decisions smooth, not jumpy")

print("\n🔮 FOR MNEMOSYNE:")
print("When you search for code:")
print("1. Compute dot products (similarity scores)")
print("2. Apply sigmoid (convert to confidence)")
print("3. Rank by confidence")
print("4. Show top results to user")
print("5. These scores help the LLM know which context to trust")
print("=" * 60)