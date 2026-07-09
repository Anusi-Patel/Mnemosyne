import os
from pathlib import Path
import chromadb
from groq import Groq
import streamlit as st

# ==========================================
# 1. UI Setup
# ==========================================
st.set_page_config(page_title="Mnemosyne", page_icon="🧠", layout="wide")
st.title("🧠 Mnemosyne: The Titan of Memory")

# ==========================================
# 2. State & Engine Initialization
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# @st.cache_resource ensures the database doesn't restart every time you type a message
@st.cache_resource 
def init_engines():
    llm = Groq(api_key=GROQ_API_KEY)
    db = chromadb.Client()
    try:
        db.delete_collection("mnemosyne_ui")
    except:
        pass
    collection = db.create_collection(name="mnemosyne_ui")
    return llm, collection

llm_client, collection = init_engines()

# Setup Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False

# ==========================================
# 3. Sidebar (Control Panel)
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    folder_path = st.text_input("Directory to Index:", value="my_code")
    
    if st.button("Index Codebase"):
        with st.spinner("Scanning and reading files..."):
            python_files = list(Path(folder_path).rglob("*.py"))
            
            if not python_files:
                st.error("No Python files found in that directory!")
            else:
                documents, metadatas, ids = [], [], []
                chunk_counter = 0
                
                for file_path in python_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.read().split('\n')
                        
                        # 50-line chunks
                        for i in range(0, len(lines), 50):
                            chunk_text = '\n'.join(lines[i:i+50])
                            if chunk_text.strip():
                                documents.append(chunk_text)
                                metadatas.append({"file": file_path.name})
                                ids.append(f"chunk_{chunk_counter}")
                                chunk_counter += 1
                    except Exception as e:
                        st.warning(f"Skipped {file_path.name}: {e}")
                
                if documents:
                    collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    st.session_state.is_indexed = True
                    st.success(f"Successfully memorized {len(documents)} code chunks!")

# ==========================================
# 4. Main Chat Interface
# ==========================================
if not st.session_state.is_indexed:
    st.info("👈 Please enter a folder path in the sidebar and click 'Index Codebase' to begin.")
else:
    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box
    if user_query := st.chat_input("Ask a question about your codebase..."):
        
        # 1. Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching the codebase..."):
                # Retrieve top 3 relevant chunks
                results = collection.query(query_texts=[user_query], n_results=3)
                
                if not results['documents'][0]:
                    st.markdown("I couldn't find anything relevant.")
                else:
                    # Combine context from the 3 chunks
                    context = ""
                    sources = set()
                    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                        context += f"\n--- File: {meta['file']} ---\n{doc}\n"
                        sources.add(meta['file'])
                    
                    # Augment
                    system_prompt = f"""
                    You are Mnemosyne, an expert coding assistant. 
                    Answer the user's question based ONLY on the provided code context.
                    If the answer isn't in the context, say so. Do not guess.
                    
                    CONTEXT:
                    {context}
                    """
                    
                    # Generate via Groq
                    chat_completion = llm_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        model="llama-3.3-70b-versatile", 
                    )
                    
                    response_text = chat_completion.choices[0].message.content
                    source_footer = f"\n\n*(Sourced from: {', '.join(sources)})*"
                    full_answer = response_text + source_footer
                    
                    st.markdown(full_answer)
                    st.session_state.messages.append({"role": "assistant", "content": full_answer})