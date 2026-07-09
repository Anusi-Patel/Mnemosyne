import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
from pathlib import Path
import chromadb
from groq import Groq

print("=" * 60)
print("MNEMOSYNE v2.0: DIRECTORY INGESTION")
print("=" * 60)

llm_client = Groq(api_key=GROQ_API_KEY)

# 2. Setup Memory Bank
chroma_client = chromadb.Client()
# We reset the database on startup so we don't duplicate old files
try:
    chroma_client.delete_collection("mnemosyne_codebase")
except:
    pass
collection = chroma_client.create_collection(name="mnemosyne_codebase")

# 3. The Chunker (Reads your hard drive)
def index_codebase(directory_path):
    print(f"\n📂 Scanning directory: {directory_path}...")
    # Find all Python files
    python_files = list(Path(directory_path).rglob("*.py"))
    
    if not python_files:
        print("❌ No Python files found! Did you create the folder?")
        return False

    documents = []
    metadatas = []
    ids = []
    chunk_counter = 0

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Chop the file into 50-line chunks
            lines = content.split('\n')
            chunk_size = 50
            
            for i in range(0, len(lines), chunk_size):
                chunk_text = '\n'.join(lines[i:i+chunk_size])
                
                if not chunk_text.strip(): continue # Skip empty lines
                
                documents.append(chunk_text)
                metadatas.append({"file": file_path.name, "start_line": i+1})
                ids.append(f"chunk_{chunk_counter}")
                chunk_counter += 1
                
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
            
    if documents:
        print(f"🧠 Memorizing {len(documents)} chunks of code...")
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print("✓ Memory index complete!")
        return True

# 4. Run the Indexer
# Point this at the folder you just created!
success = index_codebase("my_code")

if success:
    # 5. The Chat Loop
    print("\n" + "=" * 60)
    print("Mnemosyne is ready. Ask about your code. (Type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        question = input("\n👤 You: ")
        if question.lower() == 'quit':
            break
            
        # Retrieve
        results = collection.query(query_texts=[question], n_results=10)
        
        if not results['documents'][0]:
            print("Mnemosyne couldn't find relevant code.")
            continue
            
        retrieved_code = results['documents'][0][0]
        file_name = results['metadatas'][0][0]['file']
        
        # Augment
        prompt = f"""
        You are Mnemosyne, an expert coding assistant. 
        Answer the user's question based ONLY on the provided code context.
        
        Context (from {file_name}):
        ```python
        {retrieved_code}
        ```
        
        User Question: {question}
        """
        
        # Generate
        chat_completion = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile", 
        )
        
        print("\n✨ MNEMOSYNE ✨")
        print(f"(Sourced from {file_name})")
        print(chat_completion.choices[0].message.content)