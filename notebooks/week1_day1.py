import numpy as np

print("=" * 60)
print("MNEMOSYNE - WEEK 1, DAY 1: Vectors and Arrays")
print("The Titan Awakens: Understanding the Foundation")
print("=" * 60)

# Exercise 1: Understanding Arrays vs Lists
print("\n=== Exercise 1: Arrays vs Lists ===")
print("Understanding the difference between Python lists and NumPy arrays\n")

# Python list
python_list = [1, 2, 3, 4, 5]
print(f"Python list: {python_list}")
print(f"List * 5: {python_list * 5}")
print("^ Notice: It repeats the list 5 times")

# NumPy array
numpy_array = np.array([1, 2, 3, 4, 5])
print(f"\nNumPy array: {numpy_array}")
print(f"Array * 5: {numpy_array * 5}")
print("^ Notice: It multiplies each element by 5 (vectorized operation!)")

# Exercise 2: Creating Arrays
print("\n" + "=" * 60)
print("=== Exercise 2: Different Ways to Create Arrays ===")
print("NumPy gives us many tools to create arrays efficiently\n")

# Create an array of zeros (size 5)
zeros = np.zeros(5)
print(f"Zeros array: {zeros}")

# Create an array of ones (size 5)  
ones = np.ones(5)
print(f"Ones array: {ones}")

# Create an array from 0 to 9
range_array = np.arange(10)
print(f"Range array (0 to 9): {range_array}")

# Create a random array (5 elements, between 0 and 1)
random_array = np.random.rand(5)
print(f"Random array: {random_array}")
print("^ Each time you run this, these numbers will be different!")

# Exercise 3: Array Operations
print("\n" + "=" * 60)
print("=== Exercise 3: Basic Operations ===")
print("NumPy lets us do math on entire arrays at once\n")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"Array a: {a}")
print(f"Array b: {b}")

# Add them
sum_ab = a + b
print(f"\nSum (a + b): {sum_ab}")
print("^ Adds element-wise: [1+4, 2+5, 3+6]")

# Multiply them (element-wise)
product_ab = a * b
print(f"Product (a * b): {product_ab}")
print("^ Multiplies element-wise: [1*4, 2*5, 3*6]")

# Find the mean of a
mean_a = np.mean(a)
print(f"\nMean of a: {mean_a}")
print(f"^ Average: (1+2+3)/3 = {mean_a}")

# Exercise 4: Your First "AI" Calculation
print("\n" + "=" * 60)
print("=== Exercise 4: Weighted Sum (Baby Neuron!) ===")
print("This is EXACTLY what happens inside a neural network!\n")

# Imagine these are pixel values from an image
inputs = np.array([0.5, 0.3, 0.8])
print(f"Inputs (imagine these are pixel values): {inputs}")

# These are "weights" - how important each input is
weights = np.array([0.2, 0.8, 0.1])
print(f"Weights (importance of each input): {weights}")

# Calculate the weighted sum
# Method 1: Element-wise multiply, then sum
weighted_sum = np.sum(inputs * weights)

print(f"\nElement-wise multiplication: {inputs * weights}")
print(f"Weighted sum: {weighted_sum}")
print(f"^ Calculation: (0.5×0.2) + (0.3×0.8) + (0.8×0.1) = {weighted_sum}")

print("\n" + "=" * 60)
print("🎉 CONGRATULATIONS! 🎉")
print("You just computed what a neuron does in a neural network!")
print("This weighted sum is the FOUNDATION of all deep learning.")
print("=" * 60)

# Challenge: Try changing the weights and inputs above
print("\n💡 CHALLENGE: ")
print("Go back and change the 'inputs' and 'weights' values.")
print("Run the code again and see how the weighted_sum changes.")
print("This is how neural networks 'learn' - by adjusting weights!")

print("\n🔮 FOR MNEMOSYNE:")
print("When Mnemosyne searches through code, she'll use vectors")
print("to represent meaning. Similar code will have similar vectors.")
print("The dot product (which we'll learn tomorrow) measures similarity!")
print("=" * 60)