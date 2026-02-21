# Project Specification: Local PDF Retrieval-Augmented Generation (RAG) Chatbot

## 1. Project Objective
Develop a fully local, privacy-preserving chatbot that answers user queries based on provided PDF documents. The system must operate without internet access for data processing, utilizing local LLMs and vector databases.

## 2. Execution Strategy: Parallel Submodules
The project is divided into 4 independent submodules (A, B, C, D) that can be developed concurrently using mock data, followed by a final Integration Module (E). 

Agents assigned to Submodules A-D must build standalone, testable Python scripts with clear functional interfaces.

---

### Submodule A: Document Ingestion Pipeline
**Goal:** Extract, clean, and chunk text from uploaded PDFs.
* **Tasks:**
    1. Implement a function to read PDFs and extract raw text (use `PyPDF2` or `pdfplumber`).
    2. Implement a text cleaner to remove excessive whitespace, headers, and footers.
    3. Implement a chunking function using a recursive character text splitter (target: 500-1000 character chunks with 10% overlap).
* **Edge Cases:**
    * **Scanned or Image-Based PDFs:** Handle PDFs without selectable text (e.g. by integrating OCR like `pytesseract` or returning a specific error alert to the user).
    * **Encrypted/Password-Protected PDFs:** Catch extraction errors from locked PDFs and return a user-friendly error message.
    * **Corrupted or Masquerading Files:** Validate the file's mime-type before attempting to parse it to prevent parsing errors from non-PDF files renamed with a `.pdf` extension.
    * **Massive Documents:** Impose a reasonable file size or page count limit to prevent excessive memory usage or extremely long processing times.
* **Expected Input:** Filepath to a `.pdf` file.
* **Expected Output:** A Python list of string chunks.
* **Testing:** Create a mock PDF, run the pipeline, and assert that the output is a list of strings of the correct length.

### Submodule B: Vector Database & Embeddings
**Goal:** Convert text chunks into vector embeddings and store them for rapid similarity search.
* **Tasks:**
    1. Initialize a local instance of `ChromaDB` (persistent local storage). Use unique collection names (e.g., based on session ID or timestamp) to ensure concurrent user safety.
    2. Connect to a local embedding model (use `Ollama` with `nomic-embed-text` or `HuggingFaceBgeEmbeddings`).
    3. Write an `add_documents(chunks)` function to vectorize and store text.
    4. Write a `query_database(user_query, k=3)` function to retrieve the top *k* most relevant chunks.
* **Edge Cases:**
    * **Ollama Server Offline/Missing Model:** Implement a startup health check to verify the Ollama API is reachable and the required embedding models are present.
    * **Context Contamination (Session Management):** Explicitly manage database collections per session/document to prevent answers from bleeding across different uploaded PDFs in the same app instance.
    * **Low Relevance Threshold:** Implement a distance/similarity score threshold. If the top $k$ chunks don't meet the threshold, return an empty context so the LLM responds accurately with "I don't know" rather than hallucianting based on irrelevant data.
* **Expected Input:** List of string chunks (for storage) OR string query (for retrieval).
* **Expected Output:** List of the top *k* retrieved string chunks.
* **Testing:** Feed a hardcoded list of strings into the DB, query a related concept, and assert the correct string is returned.

### Submodule C: Local LLM Integration
**Goal:** Connect to a local language model to generate answers based purely on provided context.
* **Tasks:**
    1. Set up an API connection to a local `Ollama` instance (target model: `llama3` or `mistral`).
    2. Design a strict system prompt: *"You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say 'I don't know'."*
    3. Write a `generate_response(user_query, retrieved_context)` function that returns a streaming response generator.
* **Edge Cases:**
    * **Context Window Overflow:** Implement token counting (e.g., using `tiktoken`) to ensure the combined `retrieved_context` and system prompt do not exceed the model's token limit, truncating if necessary.
* **Expected Input:** User query (string) + Retrieved context (concatenated string).
* **Expected Output:** Generated response (generator/stream yielding string chunks).
* **Testing:** Pass a hardcoded context and a matching query; assert the LLM returns a relevant string stream.

### Submodule D: User Interface
**Goal:** Build a clean web interface for user interaction and file uploading.
* **Tasks:**
    1. Create a `Streamlit` application. Use `st.session_state` to explicitly cache the vector database connection and chat history across UI re-renders.
    2. Build a sidebar with a file upload widget (restricted to `.pdf`).
    3. Build a main chat window displaying message history.
    4. Provide input fields for user questions. Prevent the user from submitting new queries while the current response is generating.
    5. *Note: Mock the backend processing functions during this phase.*
* **Edge Cases:**
    * **State Reset on Re-render:** Streamlit re-runs the script on every interaction; failing to manage session state will result in lost context or UI bugs.
    * **Long Processing Times:** UI must implement visual blocking indicators (e.g., `st.spinner` or `st.status`) during "Ingesting PDF", "Embedding", and "Generating Response" states.
* **Expected Input:** User clicks, file uploads, text input.
* **Expected Output:** Interactive UI rendering chat history and processing states.

---

### Submodule E: Orchestration (Sequential Integration)
**Goal:** Connect Submodules A, B, C, and D into a single cohesive application.
* **Execution Order:** This module can only be executed once A-D are complete.
* **Tasks:**
    1. Wire the UI (Submodule D) file uploader to the Ingestion Pipeline (Submodule A).
    2. Pipe the output of Submodule A into the Vector DB (Submodule B).
    3. Wire the UI chat input to trigger Vector DB retrieval (Submodule B).
    4. Pass the retrieved context and user query to the LLM (Submodule C).
    5. Stream the LLM output back to the UI (Submodule D) using iterative UI elements for real-time generation typing effects.

## 3. Tech Stack Requirements
* **Language:** Python 3.10+
* **Frameworks:** LangChain, Streamlit
* **Local Models:** Ollama (LLM and Embeddings)
* **Database:** ChromaDB
