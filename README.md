# 🧠 Mnemosyne

Mnemosyne is a RAG-powered chat assistant for your codebase. Point it at a local
folder or a public GitHub repo, and it indexes the code into a vector database so
you can ask questions about it in plain English — and even ask it to write new
files based on what it's learned.

Named after the Greek Titan of memory.

## Features

- **Chat with your codebase** — index a local folder or clone a public GitHub
  repo, then ask questions and get answers grounded in the actual source.
- **Multi-language support** — indexes Python, JS/TS/JSX/TSX, Java, Go, Rust,
  C/C++, Markdown, and HTML.
- **Agentic file writing** — ask it to build something (e.g. *"create a file
  called `utils/slugify.js` that slugifies a string"*) and it writes real files
  to disk.
- **Secrets-aware indexing** — automatically skips `.env` files, credentials,
  private keys, and anything with "secret"/"credential"/"private_key" in the
  filename, so they never end up in the vector store or get sent to the LLM.
- **Conversation memory** — full chat history is passed to the model so it
  remembers earlier questions in the session.

## How it works

1. You point Mnemosyne at a folder (local path or GitHub URL).
2. It walks the directory, skips dependency folders (`node_modules`, `.git`,
   `venv`, etc.) and sensitive files, then chunks each supported file into
   30-line pieces.
3. Chunks are embedded and stored in a [ChromaDB](https://www.trychroma.com/)
   collection.
4. When you ask a question, the top 5 most relevant chunks are retrieved and
   passed to [Groq](https://groq.com/)'s `llama-3.3-70b-versatile` as context.
5. The model answers your question, citing sources — or, if you asked it to
   build something, emits `[FILE: path]` blocks that Mnemosyne writes to disk.

## Project structure

```
app.py                  # the main Streamlit app — start here
requirements.txt
.streamlit/
  secrets.toml.example  # copy to secrets.toml and add your API key

my_code/                # an earlier iteration of the app, kept for reference
sample_code/            # dummy files used to test the indexing pipeline
notebooks/               ┐
genesis.py               │  the step-by-step build-up: learning notes on
real_memory.py           │  embeddings, vector search, and RAG concepts
mnemsearch.py            │  that led to the final app
mnemosyne_core.py        ┘
```

`app.py` is the app to run. Everything else documents the journey of building
it, from first-principles vector math up to the full Streamlit UI.

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/<your-username>/Mnemosyne.git
cd Mnemosyne
pip install -r requirements.txt
```

**2. Add your Groq API key**

Get a free key at [console.groq.com](https://console.groq.com/keys), then
either:

- Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and
  paste in your key, **or**
- Set it as an environment variable:
  ```bash
  export GROQ_API_KEY="your-key-here"
  ```

**3. Run the app**

```bash
streamlit run app.py
```

## Usage

1. In the sidebar, choose **Local Folder** or **GitHub URL**.
2. Point it at the code you want to index and click **Index Codebase** /
   **Clone and Index**.
3. Ask questions in the chat, e.g.:
   - *"Where is the login logic handled?"*
   - *"Create a file called `src/utils/validate.js` that validates an email address"*

## Tech stack

- [Streamlit](https://streamlit.io/) — UI
- [ChromaDB](https://www.trychroma.com/) — vector storage
- [Groq](https://groq.com/) — LLM inference (`llama-3.3-70b-versatile`)

## Notes / limitations

- Only public GitHub repos can be cloned via the GitHub URL tab.
- The vector DB is in-memory and rebuilt each time you index — nothing
  persists between sessions.
- This is a learning/portfolio project, not hardened for production use.

## License

Add a license of your choice (e.g. MIT) if you want others to be able to reuse
this code — see [choosealicense.com](https://choosealicense.com/).
