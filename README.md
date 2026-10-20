chat.py 
-------
POST /api/chat/
        ↓
ChatRequest validation
        ↓
get_rag_pipeline()
        ↓
Shared RAGPipeline
        ↓
Thread Pool
        ↓
rag_pipeline.execute()
        ↓
FAISS + BM25 + RRF
        ↓
Cross-Encoder
        ↓
Groq
        ↓
ChatResponse

----------------------------------------------

# Task: Create a Complete, Professional README.md for My Multi-PDF RAG Project

Act as a senior AI engineer, Python developer, technical writer, and open-source documentation specialist.

I have built a project called **Multi-PDF RAG — Hybrid RAG Assistant**. I want you to inspect my existing project repository and generate a comprehensive, professional, end-to-end `README.md` suitable for GitHub, recruiters, developers, and technical reviewers.

**Important: Inspect the actual source code before writing the README. Do not invent features, endpoints, dependencies, commands, test results, or architecture details.**

## 1. Understand My Project

The application allows users to upload PDF documents, process and index their contents, ask questions across the indexed documents, and receive answers with supporting source information.

The project combines dense vector retrieval and sparse keyword retrieval, followed by hybrid fusion, reranking, and LLM-based answer generation.

The application has two frontend options:

- **Streamlit frontend:** A landing page, Home page, PDF Documents page, and RAG Chat page.
- **HTML frontend:** HTML templates served by the FastAPI backend.

The backend is implemented using FastAPI.

Use these details as starting context, but verify the actual implementation from the repository.

## 2. Inspect the Repository First

Inspect the project structure and the relevant files before generating documentation.

Prioritize the following locations where present:

- `backend/main.py`
- `backend/routes/`
- `backend/schemas/`
- `config/settings.py`
- `src/ingestion/`
- `src/embeddings/`
- `src/retrieval/`
- `src/orchestration/`
- `src/generation/`
- `src/utils/`
- `st_app/app.py`
- `st_app/pages/home.py`
- `st_app/pages/pdf.py`
- `st_app/pages/chat.py`
- `web_app/templates/`
- `web_app/static/`
- `data/`
- `requirements.txt`
- `.env.example`, if present
- `.gitignore`
- Any tests, configuration files, deployment files, or additional modules that materially affect how the application works.

Inspect the actual implementations of the ingestion pipeline, PDF loader, metadata manager, dense index, sparse index, hybrid fusion, retriever, reranker, query processor, session memory, prompt builder, and LLM client.

Determine how the Streamlit and HTML frontends communicate with the backend and what data structures each API endpoint expects and returns.

Do not assume the project tree listed here is complete. Generate the final structure from the files that actually exist.

## 3. README Structure

Create a polished README with the following sections, adapting them to the actual implementation.

### A. Project Title and Overview

Use the project name:

**Multi-PDF RAG — Hybrid RAG Assistant**

Write a concise introduction explaining:

- The problem the project solves.
- Why querying multiple PDF documents is useful.
- How retrieval-augmented generation is used.
- How users upload documents and ask questions.
- How retrieval results and source information support grounded answers.

Do not claim that all generated answers are guaranteed to be correct.

### B. Key Features

Document the features that are implemented, such as:

- Multi-PDF ingestion.
- PDF text extraction.
- Text cleaning and chunking.
- Document metadata and registry management.
- Dense vector search using FAISS.
- Sparse keyword search using BM25.
- Reciprocal Rank Fusion (RRF) for hybrid retrieval.
- Cross-encoder reranking.
- Query processing or contextual query rewriting, if implemented.
- LLM-based answer generation using Groq.
- Session-based conversational memory.
- Source metadata displayed in the chat interface.
- PDF upload and document listing.
- FastAPI REST endpoints.
- Streamlit and HTML frontends.
- Persistent indexes for permanent documents.
- Temporary upload cleanup on server restart, if the implemented code correctly supports it.

Verify each feature before listing it. Distinguish implemented functionality from planned functionality.

### C. Architecture

Explain the end-to-end data flow using a Mermaid flowchart.

Show the ingestion flow:

PDF Upload → PDF Loading → Text Cleaning → Chunking → Metadata Enrichment → Embeddings → FAISS Index

Show how chunks also feed the BM25 index.

Show the query flow:

User Question → Query Processing → Dense Retrieval + Sparse Retrieval → Reciprocal Rank Fusion → Cross-Encoder Reranking → Prompt Construction → Groq LLM → Answer and Sources

Include the FastAPI backend and both frontend options in the architecture where appropriate.

Use the correct component names from the code. Do not assume that embeddings are generated at a particular stage unless the source code confirms it.

### D. Technology Stack

Create a concise Markdown table covering the technologies actually used.

Investigate and document, where applicable:

- Python
- FastAPI
- Streamlit
- HTML, CSS and JavaScript
- FAISS
- BM25 / `rank_bm25`
- Hugging Face or FastEmbed-related embedding components
- LangChain document abstractions
- Cross-encoder reranking
- Groq API
- Pydantic Settings
- Requests and other relevant dependencies.

Verify the libraries and their actual roles from the imports, configuration, and implementation.

Document configured model names accurately. The settings have referenced `BAAI/bge-small-en-v1.5` for embeddings, `Xenova/ms-marco-MiniLM-L-6-v2` for reranking, and `openai/gpt-oss-120b` as the Groq model. Verify these against the current settings before publishing them.

Do not invent library versions. Obtain versions from dependency files or explicitly label them as unverified.

### E. Repository Structure

Generate a clean project tree from the actual repository.

Explain the purpose of important directories and modules. Do not list nonexistent files as if they exist.

Keep the explanations concise enough that readers can quickly understand where ingestion, retrieval, generation, API routes, frontend pages, configuration, and data storage are implemented.

### F. Installation and Setup

Write reproducible instructions for a developer using Windows and Command Prompt, matching my current development environment.

Cover:

1. Cloning or downloading the repository.
2. Opening the project directory.
3. Creating a virtual environment.
4. Activating the virtual environment.
5. Installing dependencies using the actual dependency file.
6. Configuring environment variables.
7. Setting up the required API key.
8. Launching the FastAPI backend.
9. Launching Streamlit in a separate terminal.
10. Opening the available local application URLs.

Use clear command blocks.

The current local development setup has used:

- Project directory: `Multi_PDF_Rag`
- Backend URL: `http://127.0.0.1:8000`
- FastAPI Swagger documentation: `http://127.0.0.1:8000/docs`
- Streamlit: `http://localhost:8501`

Verify these details against the current code.

Use correct working directories, particularly when starting Streamlit from `st_app`.

### G. Environment Variables

Inspect `config/settings.py`, `.env` handling, and any API clients.

Create a table with:

- Variable name.
- Purpose.
- Whether it is required.
- Default value, if present in the code.

Include the Groq API key and configured model settings where appropriate.

Provide a safe `.env.example` only if useful, and never expose real secrets, API keys, tokens, or personal credentials.

Do not put actual credentials in the README.

Clarify whether the application can initialize without the API key and what functionality would fail.

### H. How the Application Works

Explain the major pipeline stages in technically accurate but readable language:

1. PDF discovery and loading.
2. Text extraction.
3. Cleaning and chunking.
4. Document registration and metadata.
5. Embedding generation.
6. Dense retrieval.
7. Sparse retrieval.
8. Hybrid result fusion using RRF.
9. Reranking.
10. Prompt construction.
11. LLM answer generation.
12. Source metadata and response rendering.
13. Conversational context management.

Explain the difference between dense semantic retrieval and sparse keyword retrieval.

Describe index persistence, loading, rebuilding, and startup behavior according to the actual code.

### I. Document Storage and Lifecycle

Explain the actual data locations and lifecycle.

The project has used:

- `data/raw_pdfs/` for permanent PDFs.
- `data/temp_uploads/` for temporary uploaded PDFs.
- `data/document_registry.json` for document metadata.
- `data/vector_store/faiss_index/` for FAISS index files and metadata.
- `data/vector_store/bm25_index/` for BM25 index files and metadata.

Verify these locations against `config/settings.py`, the ingestion pipeline, and the startup cleanup implementation.

Explain:

- Which files are considered permanent.
- Where temporary uploads are stored.
- Whether temporary files are removed on backend restart.
- How document registry records are updated or removed.
- How FAISS and BM25 indexes are rebuilt.
- What happens when there are no permanent PDFs.
- Any important limitations involving duplicate files, old registry records, or failed index rebuilding.

Be especially precise here. Do not state that temporary cleanup is fully reliable unless the code supports that claim.

### J. API Reference

Inspect every registered API route and document the actual HTTP method, URL, request format, response format, and likely error responses.

Known endpoints to verify include:

- `GET /api/health/`
- `GET /api/documents/`
- `POST /api/documents/upload`
- `GET /api/documents/{doc_id}`
- `POST /api/chat/`

Check route prefixes and slash behavior in the actual router definitions.

Include realistic request examples for uploading a PDF and sending a chat question.

For JSON endpoints, show the exact field names accepted by the backend. For multipart uploads, document the correct field name.

Do not guess response field names. Use the schemas and route implementations as the source of truth.

### K. Frontend Usage

Document both interfaces.

**Streamlit:**
- Landing page.
- Home page.
- PDF document listing and upload.
- RAG Chat.
- New-chat behavior.
- How the frontend connects to FastAPI.
- The relevant environment variable for configuring the backend URL, if supported.

**HTML frontend:**
- The templates and route URLs.
- Static asset handling.
- Favicon location, if configured.
- How it communicates with FastAPI.

Clearly distinguish available features from UI elements that are visual-only or not yet implemented.

### L. Retrieval and Configuration Parameters

Explain configured settings such as:

- Chunk size and overlap.
- Dense retrieval top-k.
- Sparse retrieval top-k.
- Reranking top-n.
- RRF smoothing constant.
- LLM temperature and maximum tokens.

Use the actual defaults in `config/settings.py`.

Briefly explain how these settings influence retrieval quality, context size, or response generation.

### M. Troubleshooting

Include practical solutions for errors supported by the project, including:

- Missing Groq API key.
- FastAPI backend not reachable.
- Streamlit cannot connect to FastAPI.
- Missing or invalid FAISS index.
- Missing or invalid BM25 index.
- PDF upload failure.
- PDF extraction yielding no text.
- Duplicate or stale document entries.
- Index rebuilding failure.
- Python import or dependency errors.
- Port 8000 or 8501 already in use.
- Differences between permanent and temporary PDFs.

Don't invent errors as confirmed issues. Present them as troubleshooting possibilities where appropriate.

### N. Limitations and Future Improvements

Identify genuine limitations after examining the code. Clearly distinguish existing limitations from suggestions.

Potential areas to assess include:

- Persistent storage of conversation histories.
- Authentication and user isolation.
- Production CORS configuration.
- PDF deletion and index consistency.
- Upload validation and file-size limits.
- Retrieval evaluation and testing.
- Background ingestion.
- Deployment and environment-specific configuration.

Do not claim that a feature is missing without checking the implementation first.

### O. Testing

Inspect existing tests and testing instructions.

Document how to run tests if they exist. If there is no test suite, state that accurately and provide a small, clearly labeled manual smoke-test checklist if appropriate.

Do not fabricate test coverage, benchmark results, evaluation metrics, or successful test runs.

### P. Screenshots

Create a small screenshots section with paths only for screenshots that actually exist in the repository. Otherwise, add clearly labeled placeholders describing which screenshots I can add later.

Useful screenshots could include:

- Landing page.
- PDF document dashboard.
- Chat interaction showing an answer and source references.

### Q. License and Contributions

Inspect the repository for an existing license. If one exists, document it accurately. If none exists, do not assume a license or insert a claim that the project is open-source under a particular license.

Add a concise contribution section appropriate to the repository.

## 4. Writing and Formatting Standards

The README must:

- Be written in professional, natural English.
- Use GitHub-compatible Markdown.
- Include a clickable table of contents.
- Use clear heading levels.
- Use syntax-highlighted code blocks.
- Use Mermaid diagrams where useful.
- Use Markdown tables for configuration and API documentation.
- Be technically detailed but easy to navigate.
- Explain technical concepts without unnecessary repetition.
- Avoid generic marketing claims.
- Avoid excessive emojis, decorative badges, and unsupported claims.
- Be appropriate for a portfolio project reviewed by AI engineers, backend developers, and recruiters.

Do not add badges for CI, coverage, releases, or licensing unless the repository verifies that they exist.

## 5. Accuracy and Verification Rules

Before writing, inspect the actual source files and cross-check:

- Imports against dependency declarations.
- API endpoints against router definitions.
- Configuration defaults against settings.
- Directory paths against the filesystem configuration.
- Upload logic against the ingestion pipeline.
- Startup cleanup against the index-building flow.
- UI links against the routes implemented in the backend.

If two files disagree, do not silently choose one version. Explain the discrepancy briefly in your internal assessment, then document only the behavior that can be established from the implementation.

Do not reveal secrets or include `.env` contents.

Do not modify application code, delete files, or regenerate indexes as part of this task.

## 6. Deliverable

Create a complete, ready-to-commit `README.md` in the root directory of the existing repository.

Do not give me only an outline or a list of suggested sections. Produce the actual Markdown file.

After creating it:

1. Verify that all local file paths and commands in the README match the actual repository.
2. Check the Mermaid syntax.
3. Check that code blocks and tables are properly formatted.
4. Ensure the documented environment variables and API routes are accurate.
5. Summarize the important files and commands I should review before committing the README.

If any detail cannot be verified from the available repository, mark it explicitly as needing confirmation rather than inventing an answer.