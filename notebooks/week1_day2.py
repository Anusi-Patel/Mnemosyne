import numpy as np

print("=" * 60)
print("MNEMOSYNE - WEEK 1, DAY 2: Matrices and Shapes")
print("Building the Architecture: Understanding Dimensions")
print("=" * 60)

# Exercise 1: Understanding Shapes
print("\n=== Exercise 1: What is Shape? ===")
print("Shape tells us the dimensions of our data\n")

# A vector (1D)
vector = np.array([1, 2, 3, 4, 5])
print(f"Vector: {vector}")
print(f"Shape: {vector.shape}")
print(f"^ This means: 5 elements in 1 dimension\n")

# A matrix (2D)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(f"Matrix:\n{matrix}")
print(f"Shape: {matrix.shape}")
print(f"^ This means: 2 rows, 3 columns (2x3 matrix)\n")

# A 3D array (tensor)
tensor = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(f"3D Tensor:\n{tensor}")
print(f"Shape: {tensor.shape}")
print(f"^ This means: 2 matrices, each 2x2")
print("(Images in AI are often stored like this: height x width x color_channels)")

# Exercise 2: Creating Matrices
print("\n" + "=" * 60)
print("=== Exercise 2: Creating Matrices ===")
print("Different ways to create 2D arrays\n")

# Matrix of zeros
zeros_matrix = np.zeros((3, 4))  # 3 rows, 4 columns
print(f"3x4 matrix of zeros:\n{zeros_matrix}\n")

# Matrix of ones
ones_matrix = np.ones((2, 5))  # 2 rows, 5 columns
print(f"2x5 matrix of ones:\n{ones_matrix}\n")

# Random matrix
random_matrix = np.random.rand(3, 3)  # 3x3 random values
print(f"3x3 random matrix:\n{random_matrix}\n")

# Identity matrix (1s on diagonal, 0s elsewhere - very important!)
identity = np.eye(4)  # 4x4 identity matrix
print(f"4x4 identity matrix:\n{identity}")
print("^ Identity matrix is like multiplying by 1 - it doesn't change things")

# Exercise 3: Accessing Elements
print("\n" + "=" * 60)
print("=== Exercise 3: Accessing Matrix Elements ===")
print("How to get specific values from a matrix\n")

sample = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print(f"Our matrix:\n{sample}\n")

# Access single element
print(f"Element at row 0, column 1: {sample[0, 1]}")
print(f"Element at row 2, column 2: {sample[2, 2]}\n")

# Access entire row
print(f"Entire row 1: {sample[1]}")
print(f"^ Gets [40, 50, 60]\n")

# Access entire column
print(f"Entire column 2: {sample[:, 2]}")
print(f"^ Gets [30, 60, 90]")
print("The ':' means 'all rows'")

# Exercise 4: Reshaping - The Most Important Skill
print("\n" + "=" * 60)
print("=== Exercise 4: Reshaping ===")
print("Changing the shape WITHOUT changing the data\n")

original = np.arange(12)  # Creates [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print(f"Original array (12 elements): {original}")
print(f"Shape: {original.shape}\n")

# Reshape to 3x4 matrix
reshaped_3x4 = original.reshape(3, 4)
print(f"Reshaped to 3x4:\n{reshaped_3x4}")
print(f"Shape: {reshaped_3x4.shape}\n")

# Reshape to 4x3 matrix
reshaped_4x3 = original.reshape(4, 3)
print(f"Reshaped to 4x3:\n{reshaped_4x3}")
print(f"Shape: {reshaped_4x3.shape}\n")

# Reshape to 2x6 matrix
reshaped_2x6 = original.reshape(2, 6)
print(f"Reshaped to 2x6:\n{reshaped_2x6}")
print(f"Shape: {reshaped_2x6.shape}\n")

print("⚠️  RULE: Total number of elements must stay the same!")
print("12 elements can become 3x4, 4x3, 2x6, 6x2, 12x1, etc.")
print("But NOT 3x5 (that would need 15 elements)")

# Exercise 5: Matrix Operations
print("\n" + "=" * 60)
print("=== Exercise 5: Matrix Math ===")
print("Operating on entire matrices at once\n")

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(f"Matrix A:\n{A}\n")
print(f"Matrix B:\n{B}\n")

# Element-wise addition
print(f"A + B:\n{A + B}\n")

# Element-wise multiplication
print(f"A * B (element-wise):\n{A * B}\n")

# Transpose (flip rows and columns)
print(f"A transpose:\n{A.T}")
print(f"^ Rows become columns, columns become rows")

# Exercise 6: Neural Network Layer Simulation
print("\n" + "=" * 60)
print("=== Exercise 6: Baby Neural Network Layer ===")
print("This is EXACTLY how a layer in a neural network works!\n")

# Imagine 4 data samples, each with 3 features
inputs = np.array([
    [0.5, 0.3, 0.8],  # Sample 1
    [0.2, 0.9, 0.1],  # Sample 2
    [0.7, 0.4, 0.6],  # Sample 3
    [0.1, 0.8, 0.3]   # Sample 4
])

print(f"Inputs (4 samples, 3 features each):\n{inputs}")
print(f"Shape: {inputs.shape} -> 4 samples × 3 features\n")

# Weights for transforming 3 features into 2 outputs
weights = np.random.rand(3, 2)  # 3 inputs -> 2 outputs
print(f"Weights (3 features -> 2 outputs):\n{weights}")
print(f"Shape: {weights.shape} -> 3 × 2\n")

# Matrix multiplication (we'll learn this deeply tomorrow!)
# For now, just know this is how data flows through a network
outputs = np.dot(inputs, weights)

print(f"Outputs (4 samples, 2 features each):\n{outputs}")
print(f"Shape: {outputs.shape} -> 4 samples × 2 features")

print("\n🎉 CONGRATULATIONS! 🎉")
print("You just passed data through a neural network layer!")
print("Input shape (4, 3) × Weights (3, 2) = Output (4, 2)")
print("=" * 60)

print("\n💡 KEY INSIGHT:")
print("Neural networks are just MATRICES multiplying MATRICES!")
print("Understanding shapes is 80% of debugging deep learning code.")

print("\n🔮 FOR MNEMOSYNE:")
print("When Mnemosyne processes code, she'll use matrices to represent:")
print("- Each line of code as a vector (row in a matrix)")
print("- Multiple files as a 3D tensor")
print("- Transforming code embeddings from one space to another")
print("Shapes tell you if your data pipeline is correct!")
print("=" * 60)