import os
import tempfile
import streamlit as st

from pdf_processor import process_pdf, PDFProcessingError
from vector_db import SessionVectorDB, VectorDBError
from llm_integration import generate_response, LLMIntegrationError

st.set_page_config(page_title="Local PDF RAG Chatbot", layout="wide")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    try:
        st.session_state.vector_db = SessionVectorDB(run_health_check=True)
    except VectorDBError as e:
        st.error(f"Failed to initialize Vector Database: {e}")
        st.stop()

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Sidebar for Upload (Submodule D)
with st.sidebar:
    st.title("📄 PDF Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file and not st.session_state.pdf_processed:
        # Simulate Submodule E integration steps dynamically via UI components
        with st.status("Processing Document...", expanded=True) as status:
            tmp_path = None
            try:
                # 1. Save uploaded file to temp
                st.write("Saving temporary file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # 2. Extract and Chunk (Submodule A -> D/E)
                st.write("Extracting and chunking text (Submodule A)...")
                chunks = process_pdf(tmp_path)
                
                # 3. Vectorize and Store (Submodule B -> D/E)
                st.write(f"Vectorizing {len(chunks)} chunks (Submodule B)...")
                
                progress_bar = st.progress(0.0)
                def ui_progress(pct):
                    progress_bar.progress(pct)
                    
                st.session_state.vector_db.add_documents(chunks, progress_callback=ui_progress)
                progress_bar.empty()
                
                st.session_state.pdf_processed = True
                status.update(label="Process Complete!", state="complete", expanded=False)
                st.success("Document ingested successfully.")
                
            except PDFProcessingError as pe:
                status.update(label="Processing Failed", state="error")
                st.error(f"PDF Error: {pe}")
            except Exception as e:
                status.update(label="Processing Failed", state="error")
                st.error(f"Unknown Error: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    if st.button("Clear Session / Upload New PDF", disabled=not st.session_state.pdf_processed):
        st.session_state.vector_db.clear_session()
        st.session_state.vector_db = SessionVectorDB(run_health_check=False)
        st.session_state.pdf_processed = False
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface (Submodule D)
st.title("🤖 Local PDF RAG Chatbot")

if not st.session_state.pdf_processed:
    st.info("Please upload a PDF document in the sidebar to begin.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input (Submodule D & E)
    if prompt := st.chat_input("Ask a question about your document..."):
        # Display Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response (Submodule C & E)
        with st.chat_message("assistant"):
            try:
                # Query DB for Context
                with st.spinner("Searching document for context..."):
                    context_chunks = st.session_state.vector_db.query_database(prompt)
                
                if not context_chunks:
                    fallback_msg = "I don't know the answer, as no relevant context was found in the document."
                    st.markdown(fallback_msg)
                    st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
                else:
                    # Stream LLM Response
                    response_stream = generate_response(prompt, context_chunks)
                    full_response = st.write_stream(response_stream)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
            except VectorDBError as ve:
                st.error(f"Vector DB Error: {ve}")
            except LLMIntegrationError as le:
                st.error(f"LLM Error: {le}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
