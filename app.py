import os
import re
from pathlib import Path
import chromadb
from groq import Groq
import google.generativeai as genai
import streamlit as st

# ==========================================
# UPGRADE NOTES (read before editing):
# 1. SMARTER RAG   — retrieves 5 chunks (was 3), uses 30-line chunks (was 50)
#                    for finer-grained matches. Full chat history is sent to
#                    the LLM so it remembers earlier messages.
# 2. MORE FILE TYPES — indexes 12 extensions across Python, JS, TS, Java,
#                    Go, Rust, C/C++, Markdown, and HTML.
# 3. AGENTIC WRITING — LLM can tag a response with [FILE: path/name.ext]
#                    and Mnemosyne will physically create that file on disk.
# 4. SECURITY       — API key moved to st.secrets / env var. Never hardcode.
# ==========================================

st.set_page_config(page_title="Mnemosyne", page_icon="📜", layout="wide")

# ==========================================
# THEME — "The Archive": ink, parchment, and antique gold.
# Fonts: Cinzel (inscription-style display, used sparingly for the title),
# Spectral (book-like serif for body/chat text), JetBrains Mono (code).
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Spectral:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Spectral', serif; }

/* Title */
h1 {
    font-family: 'Cinzel', serif !important;
    letter-spacing: 0.04em;
    color: #e9e3d3 !important;
    border-bottom: 1px solid #b8923f55;
    padding-bottom: 0.6rem;
}
.mnemosyne-tagline {
    font-family: 'Spectral', serif;
    font-style: italic;
    color: #9aa3b0;
    margin-top: -0.8rem;
    margin-bottom: 1.4rem;
    font-size: 0.95rem;
}

/* Sidebar reads like a ledger */
section[data-testid="stSidebar"] {
    background-color: #1b212b;
    border-right: 1px solid #b8923f33;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'Cinzel', serif !important;
    color: #b8923f !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.03em;
}

/* Chat messages — inscribed tablet look */
div[data-testid="stChatMessage"] {
    background-color: #1e2530;
    border-top: 2px solid #b8923f;
    border-radius: 4px;
    padding: 0.35rem 0.9rem;
    margin-bottom: 0.6rem;
}
div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li {
    font-family: 'Spectral', serif;
    color: #e9e3d3;
    line-height: 1.55;
}
div[data-testid="stChatMessage"] code {
    font-family: 'JetBrains Mono', monospace;
    color: #7fb8b6;
    background-color: #12161d;
}

/* Buttons */
.stButton button {
    font-family: 'Cinzel', serif;
    letter-spacing: 0.03em;
    background-color: #b8923f;
    color: #161b24;
    border: none;
    border-radius: 3px;
}
.stButton button:hover {
    background-color: #cba553;
    color: #161b24;
}

/* Sources / footer — stamped seal feel */
.mnemosyne-sources {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #7fb8b6;
    border-left: 2px solid #4a7a78;
    padding-left: 0.6rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Mnemosyne")
st.markdown('<p class="mnemosyne-tagline">The Titan of Memory — ask your codebase anything, in plain English.</p>', unsafe_allow_html=True)

# ==========================================
# UPGRADE 4 — API key from environment
# Set GROQ_API_KEY in your shell or a .streamlit/secrets.toml file.
# Never paste keys directly into source code.
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Set it as an environment variable or in .streamlit/secrets.toml")
    st.stop()

# ==========================================
# BUGFIX — Embeddings via Gemini API instead of ChromaDB's default
# ChromaDB's built-in embedding function downloads an ONNX model from
# Hugging Face on first run. This silently hangs/fails on restricted
# corporate networks (same issue hit in StoryWeaver). Using Gemini's
# text-embedding-004 API instead means no local model download at all —
# just an HTTPS call, same as the chat requests already being made.
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Set it as an environment variable or in .streamlit/secrets.toml")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Chroma-compatible embedding function backed by Gemini's embedding API."""

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = []
        for text in input:
            # Cap overly long chunks defensively; our chunks are 30 lines so
            # this is just a safety net, not expected to trigger normally.
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text[:8000],
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings

# ==========================================
# UPGRADE 2 — Supported file extensions
# Previously: only .py
# Now: 12 languages + docs
# ==========================================
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",   # Python & JS/TS ecosystem
    ".java", ".go", ".rs",                  # Java, Go, Rust
    ".cpp", ".c", ".h",                     # C / C++
    ".md", ".html",                         # Docs & templates
}

# Folders to always skip — these contain dependency code, not your code
IGNORED_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".next", ".cache", "vendor",
}

# ==========================================
# SECURITY — Sensitive file blocklist
# Files matching any of these exact names OR extensions are NEVER indexed,
# even if their extension is in SUPPORTED_EXTENSIONS.
# Add your own entries to either set as needed.
# ==========================================
SENSITIVE_FILENAMES = {
    # Environment & secrets
    ".env", ".env.local", ".env.development", ".env.production",
    ".env.staging", ".env.test", ".env.example",
    # Credential & key files
    "secrets.toml", "secrets.json", "credentials.json",
    "serviceAccountKey.json", ".netrc", ".pgpass",
    # SSH & certificates
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "id_rsa.pub", "id_ed25519.pub",
    # Cloud provider configs
    ".aws", "gcloud.json",
    # Package lock files (large, not useful for Q&A)
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
}

SENSITIVE_EXTENSIONS = {
    ".pem", ".key", ".p12", ".pfx", ".cer", ".crt",  # Certificates & keys
    ".ppk",                                            # PuTTY private key
    ".jks",                                            # Java keystore
    ".sqlite", ".db",                                  # Local databases
}

def is_sensitive(file_path: Path) -> bool:
    """Return True if this file should never be indexed."""
    name = file_path.name.lower()
    suffix = file_path.suffix.lower()
    # Block by exact filename (handles dotfiles like .env)
    if name in SENSITIVE_FILENAMES:
        return True
    # Block by extension (.pem, .key, etc.)
    if suffix in SENSITIVE_EXTENSIONS:
        return True
    # Block any file whose name contains 'secret', 'credential', or 'private'
    # e.g. my_secret_config.py, firebase_credentials.json
    for keyword in ("secret", "credential", "private_key"):
        if keyword in name:
            return True
    return False

# ==========================================
# UPGRADE 3 — File writing agent
# Scans LLM output for [FILE: some/path.ext] markers.
# Everything between two such markers (or end of response) is written to disk.
# ==========================================
def extract_and_write_files(text: str, output_dir: str = ".") -> list:
    """
    Parse [FILE: path] tags from LLM output and write each block to disk.
    Returns a list of file paths that were created.

    Example LLM output it handles:
        [FILE: src/utils/auth.js]
        ```javascript
        export function login(user) { ... }
        ```
    """
    pattern = r"\[FILE:\s*(.+?)\]\s*```[a-z]*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    created = []
    for rel_path, code in matches:
        rel_path = rel_path.strip()
        # Security: prevent path traversal attacks like ../../etc/passwd
        full_path = Path(output_dir) / rel_path
        if ".." in full_path.parts:
            st.warning(f"Skipped unsafe path: {rel_path}")
            continue
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(code.strip(), encoding="utf-8")
            created.append(str(full_path))
        except Exception as e:
            st.warning(f"Could not write {rel_path}: {e}")
    return created

# ==========================================
# Database + LLM init (cached so it survives reruns)
# ==========================================
@st.cache_resource
def init_engines():
    llm = Groq(api_key=GROQ_API_KEY)
    db = chromadb.Client()
    try:
        db.delete_collection("mnemosyne_ui")
    except Exception:
        pass
    collection = db.create_collection(
        name="mnemosyne_ui",
        embedding_function=GeminiEmbeddingFunction(),
    )
    return llm, collection

llm_client, collection = init_engines()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False
if "output_dir" not in st.session_state:
    st.session_state.output_dir = "."

# ==========================================
# Helper: chunk files and load into ChromaDB
# Shared by both the local folder and GitHub tab.
# ==========================================
def _add_to_collection(found_files: list, base_dir: str = None):
    """Chunk files and add to ChromaDB. Called by both index paths.

    base_dir: if provided, file paths in metadata are shown relative to it
    so sources display as 'src/utils/auth.js' instead of a full system path.
    """
    documents, metadatas, ids = [], [], []
    chunk_counter = 0
    CHUNK_SIZE = 30
    base = Path(base_dir) if base_dir else None

    skipped_sensitive = [f for f in found_files if is_sensitive(f)]
    safe_files = [f for f in found_files if not is_sensitive(f)]

    if skipped_sensitive:
        names = sorted(set(f.name for f in skipped_sensitive))[:8]
        st.info(
            f"Skipped {len(skipped_sensitive)} sensitive file(s): {', '.join(names)}"
            + (" …and more" if len(set(f.name for f in skipped_sensitive)) > 8 else "")
        )

    for file_path in safe_files:
        try:
            # Strip the base dir so sources never leak temp paths or usernames
            if base:
                try:
                    display_path = str(file_path.relative_to(base))
                except ValueError:
                    display_path = file_path.name  # fallback if relative_to fails
            else:
                display_path = str(file_path)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().split("\n")
            for i in range(0, len(lines), CHUNK_SIZE):
                chunk = "\n".join(lines[i: i + CHUNK_SIZE])
                if chunk.strip():
                    documents.append(chunk)
                    metadatas.append({
                        "file": display_path,
                        "start_line": i + 1,
                        "language": file_path.suffix.lstrip("."),
                    })
                    ids.append(f"chunk_{chunk_counter}")
                    chunk_counter += 1
        except Exception as e:
            st.warning(f"Skipped {file_path.name}: {e}")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        st.session_state.is_indexed = True
        st.success(
            f"Indexed {len(documents)} chunks across {len(safe_files)} files "
            f"({len(set(m['language'] for m in metadatas))} languages)."
        )
    else:
        st.error("No indexable content found after filtering.")


def _index_folder(folder_path: str):
    """Scan a local directory and index all supported files."""
    with st.spinner("Scanning files…"):
        found_files = []
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for fname in files:
                fp = Path(root) / fname
                if fp.suffix in SUPPORTED_EXTENSIONS:
                    found_files.append(fp)

        if not found_files:
            st.error(
                f"No supported files found in '{folder_path}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            return

        # Pass folder_path so sources show relative to the indexed root
        _add_to_collection(found_files, base_dir=folder_path)


def _clone_and_index(repo_url: str, branch: str):
    """Clone a public GitHub repo into a temp dir, index it, then clean up."""
    import tempfile, subprocess, shutil

    # Accept both github.com/user/repo and https://github.com/user/repo
    if not repo_url.startswith("http"):
        repo_url = "https://" + repo_url

    with st.spinner(f"Cloning {repo_url} …"):
        tmp_dir = tempfile.mkdtemp(prefix="mnemosyne_")
        try:
            # --depth 1 = shallow clone, only latest commit — keeps it fast
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch,
                 repo_url, tmp_dir],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                # Branch might not exist — retry without --branch (uses repo default)
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, tmp_dir],
                    capture_output=True, text=True, timeout=120,
                )
            if result.returncode != 0:
                st.error(f"Clone failed: {result.stderr.strip()}")
                return

            found_files = []
            for root, dirs, files in os.walk(tmp_dir):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
                for fname in files:
                    fp = Path(root) / fname
                    if fp.suffix in SUPPORTED_EXTENSIONS:
                        found_files.append(fp)

            if not found_files:
                st.error("No supported files found in this repo.")
                return

            # Pass tmp_dir so sources show as 'src/foo.js' not the full temp path
            _add_to_collection(found_files, base_dir=tmp_dir)

        finally:
            # Always delete the temp dir — even if indexing crashes
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ==========================================
# Sidebar — two tabs: Local Folder / GitHub URL
# ==========================================
with st.sidebar:
    st.header("📖 Open the Archive")

    tab_local, tab_github = st.tabs(["📁 Local Folder", "🐙 GitHub URL"])

    with tab_local:
        folder_path = st.text_input("Directory to index:", value="my_code")
        output_dir = st.text_input(
            "Write new files to:", value=".",
            help="Mnemosyne will create files here when you ask it to build something."
        )
        st.session_state.output_dir = output_dir

        if st.button("Index Codebase", key="btn_local"):
            _index_folder(folder_path)

    with tab_github:
        repo_url = st.text_input(
            "Repository URL:",
            placeholder="https://github.com/user/repo",
        )
        branch = st.text_input("Branch (optional):", value="main")
        st.session_state.output_dir = "."

        if st.button("Clone and Index", key="btn_github"):
            if not repo_url.strip():
                st.error("Paste a GitHub URL first.")
            else:
                _clone_and_index(repo_url.strip(), branch.strip() or "main")

    st.divider()
    st.caption("Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))

# ==========================================
# Main chat
# ==========================================
if not st.session_state.is_indexed:
    st.info("👈 Enter a folder path in the sidebar and click 'Index Codebase' to begin.")
else:
    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if user_query := st.chat_input("Ask a question or say 'Create a file called X that does Y'…"):

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching the codebase…"):

                # UPGRADE 1 — retrieve 5 chunks (was 3) for richer context
                # Embed the query with task_type="retrieval_query" (not the
                # "retrieval_document" type used for indexed code) — Gemini
                # tunes these differently, and using the matching type
                # improves match quality for asymmetric search like this.
                query_embedding = genai.embed_content(
                    model="models/text-embedding-004",
                    content=user_query,
                    task_type="retrieval_query",
                )["embedding"]
                results = collection.query(query_embeddings=[query_embedding], n_results=5)

                context = ""
                sources = []
                if results["documents"][0]:
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                        lang = meta.get("language", "")
                        context += (
                            f"\n--- File: {meta['file']} "
                            f"(lines {meta.get('start_line', '?')}+, lang: {lang}) ---\n"
                            f"{doc}\n"
                        )
                        sources.append(meta["file"])

                # UPGRADE 3 — tell the LLM it can write files using [FILE: ...] syntax
                system_prompt = f"""You are Mnemosyne, an expert coding assistant and autonomous agent.

You have two modes:

1. ANSWER mode — if the user asks a question about the codebase, answer using
   the provided CONTEXT. Cite the file name when referencing specific code.
   If the context is incomplete, use general programming knowledge but say so.

2. BUILD mode — if the user asks you to create, write, or modify a file,
   generate the full file content using this exact format:

   [FILE: relative/path/filename.ext]
   ```language
   ... full file content here ...
   ```

   You may output multiple [FILE: ...] blocks in a single response.
   Always use relative paths (e.g. "src/components/Login.tsx", not "/home/user/...").

CODEBASE CONTEXT (top 5 most relevant chunks):
{context if context else "No relevant code found — use your general knowledge."}
"""

                # UPGRADE 1 — pass full conversation history to the LLM
                # This lets Mnemosyne remember what was said earlier in the session.
                history_for_llm = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                response = llm_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *history_for_llm,
                        {"role": "user", "content": user_query},
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,   # lower = more precise / less hallucination
                )

                response_text = response.choices[0].message.content

                # UPGRADE 3 — intercept [FILE: ...] tags and write files
                created_files = extract_and_write_files(
                    response_text,
                    output_dir=st.session_state.output_dir,
                )

                # Build footer
                footer_parts = []
                if sources:
                    unique_sources = sorted(set(sources))
                    footer_parts.append(
                        f'<div class="mnemosyne-sources">📜 Consulted: {", ".join(unique_sources)}</div>'
                    )
                if created_files:
                    file_list = "".join(f"<div>— {f}</div>" for f in created_files)
                    footer_parts.append(
                        f'<div class="mnemosyne-sources">✒️ Files inscribed:{file_list}</div>'
                    )

                full_answer = response_text
                if footer_parts:
                    full_answer += "\n\n" + "".join(footer_parts)

                st.markdown(full_answer, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer}
                )