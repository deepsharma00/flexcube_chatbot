"""
Simple Web UI for Simple JSON RAG
A clean interface to query your indexed documents using Streamlit.
"""

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path
from config import QUERIER_MODEL, QUERIER_HOST

# Directory where ui.py lives — used as cwd for subprocess
SCRIPT_DIR = Path(__file__).parent.resolve()

# ──────────────────────────────────────────────────────────────────────────
# Streamlit Page Configuration
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlexCube RAG Chat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("💬 FlexCube RAG Chatbot")
st.write("Ask questions about your FlexCube documentation")

# ──────────────────────────────────────────────────────────────────────────
# Sidebar Configuration
# ──────────────────────────────────────────────────────────────────────────

# Model catalogue: (display_label, model_id, host)
MODEL_OPTIONS = [
    ("🟢 Groq — llama-3.3-70b-versatile",  "llama-3.3-70b-versatile",  "groq"),
    ("🟢 Groq — llama-3.1-70b-versatile",  "llama-3.1-70b-versatile",  "groq"),
    ("🟢 Groq — llama-3.1-8b-instant",     "llama-3.1-8b-instant",     "groq"),
    ("🟢 Groq — mixtral-8x7b-32768",       "mixtral-8x7b-32768",       "groq"),
    ("🔵 Qwen — qwen2.5:7b (Ollama)",      "qwen2.5:7b",               QUERIER_HOST),
]
MODEL_LABELS = [m[0] for m in MODEL_OPTIONS]

# Default: pick whichever label matches the .env model, else first entry
default_idx = next(
    (i for i, m in enumerate(MODEL_OPTIONS) if m[1] == QUERIER_MODEL),
    0
)

with st.sidebar:
    st.header("⚙️ Query Model")
    st.caption("Choose the LLM used to answer your questions.")

    chosen_label = st.radio(
        "Select model",
        MODEL_LABELS,
        index=default_idx,
        label_visibility="collapsed",
    )

    # Resolve selected model + host
    chosen = next(m for m in MODEL_OPTIONS if m[0] == chosen_label)
    selected_model = chosen[1]
    host_value     = chosen[2]

    st.divider()
    st.caption(f"**Model:** `{selected_model}`")
    st.caption(f"**Host:** `{host_value}`")

# Index path is always the default (not exposed to user)
index_path = "index/"

# ──────────────────────────────────────────────────────────────────────────
# Main Chat Interface
# ──────────────────────────────────────────────────────────────────────────

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Question input
question = st.chat_input("Ask a question about the documentation...", key="question_input")

if question:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    # Process the query
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        try:
            status_placeholder.info("🔄 Searching documentation...")
            
            # Build the command
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "multi_querier.py"),
                "--index", index_path,
                "--query", question,
            ]
            
            # Always pass the chosen model and host
            cmd.extend(["--host",  host_value])
            cmd.extend(["--model", selected_model])
            
            # Force UTF-8 output so Unicode chars (→, etc.) don't crash on Windows
            run_env = os.environ.copy()
            run_env["PYTHONIOENCODING"] = "utf-8"

            # Run multi_querier.py from the project directory
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                cwd=str(SCRIPT_DIR),
                env=run_env,
            )
            
            status_placeholder.empty()
            
            if result.returncode == 0:
                # Extract the response (filter out debug info if any)
                response_text = result.stdout.strip()
                response_placeholder.markdown(response_text)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                response_placeholder.error(f"❌ Error: {error_msg}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"*Error: {error_msg}*"
                })
        
        except subprocess.TimeoutExpired:
            status_placeholder.empty()
            response_placeholder.error("⏱️ Request timed out (5 minutes). Try a simpler query.")
        
        except FileNotFoundError:
            status_placeholder.empty()
            response_placeholder.error("❌ Error: Could not find multi_querier.py. Make sure you're in the correct directory.")
        
        except Exception as e:
            status_placeholder.empty()
            response_placeholder.error(f"❌ Error: {str(e)}")

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Chat History", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("📋 View Configuration", key="config_btn"):
        st.session_state.show_config = not st.session_state.get("show_config", False)
        st.rerun()

if st.session_state.get("show_config"):
    st.json({
        "index_path": index_path,
        "host": host_value or "default (from .env)",
        "model": selected_model,
        "querier_host": QUERIER_HOST,
        "querier_model": QUERIER_MODEL
    })
