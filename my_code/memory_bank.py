import chromadb

print("=" * 50)
print("MNEMOSYNE: THE MEMORY BANK (ChromaDB)")
print("=" * 50)

# 1. Initialize the Vector Database (In-memory for now)
print("Starting ChromaDB Engine...")
chroma_client = chromadb.Client()

# 2. Create a "Collection" (Think of this as a folder for a specific GitHub repo)
collection = chroma_client.create_collection(name="mnemosyne_codebase")

# 3. Add our code to the database
# Magic: We don't even need to call the embedding model manually! 
# Chroma automatically uses the exact same MiniLM Transformer you just used.
print("Embedding and storing code chunks...")
collection.add(
    documents=[
        "def verify_password(plain_password, password_hash):\n    return hash(plain_password) == password_hash",
        "button { background-color: red; color: white; }"
    ],
    metadatas=[{"file": "auth.py"}, {"file": "styles.css"}], # We attach metadata to know where it came from
    ids=["chunk_1", "chunk_2"] # Every chunk needs a unique ID
)

print("✓ Code successfully saved to Vector Database.")

# 4. Search the Database
query = "How is user authentication handled?"
print(f"\n🔍 Searching for: '{query}'")

# Chroma automatically embeds your query and does the dot product search!
results = collection.query(
    query_texts=[query],
    n_results=1 # We only want the #1 most relevant chunk
)

print("\n--- MNEMOSYNE FOUND ---")
# Chroma returns results in lists, so we access the first item [0][0]
print(f"File: {results['metadatas'][0][0]['file']}")
print(f"Code: {results['documents'][0][0]}")