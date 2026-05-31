# learnBotRAG 🧠📚

An enterprise-grade, local Retrieval-Augmented Generation (RAG) system built with Python and LangChain. It features a scalable Factory Architecture designed to parse, chunk, and index university scripts, PDF summaries, and structured CSV catalogs into a local Vector Database for intelligent querying.

## 🚀 Key Features

* **Advanced Document Loading Factory:** Supports polymorphic loading (PDFs via `PyPDFLoader`, CSVs via `CSVLoader`) driven by file extension routing.
* **Incremental Vector Indexing (Delta Updates):** Seamlessly adds new documents to the vector database without rebuilding the entire index, drastically reducing computation time.
* **Auto-Cleanup & Synchronization:** Automatically detects and purges orphaned entries (deleted local files) from the database to prevent stale data.
* **Robust Production Error Handling:** Safely intercepts and processes edge cases like unreadable, corrupted, or encrypted PDFs without crashing the data pipeline.
* **Interactive Terminal Chat:** A fast, console-based conversational interface to query your custom knowledge base with minimal latency.

## 🛠️ Software Architecture & Design Principles

This project is built from the ground up according to industry-standard **SOLID principles** and clean code patterns:
* **Dependency Injection:** Loosely coupled core components (Loaders, Trackers, Vector Store) allowing effortless extensions or model swaps (e.g., swapping local embeddings for OpenAI/Ollama).
* **Automated Unit Testing:** Backed by an automated test suite (`pytest`) verifying parser consistency, data pipelines, and error handling behaviors.

## ⚙️ Tech Stack

* **Core Language:** Python 3.10+
* **Framework:** LangChain Ecosystem
* **Automation & Formatting:** Ruff, Black
* **Testing:** Pytest

## 🚀 Setup & Installation

1. **Clone the repository:**
```bash
   git clone [https://github.com/oussemaster/learnBotRAG.git](https://github.com/oussemaster/learnBotRAG.git)
   cd learnBotRAG
   ```

2. **Create and activate a virtual environment:**
```bash
   python3 -m venv .virtualEnvDir
   source .virtualEnvDir/bin/activate  # On Windows use: .virtualEnvDir\Scripts\activate
   ```

3. **Install dependencies:**
```bash
   pip install -r requirements.txt
   ```


2. **Environment Variables:**
Create a .env file in the root directory and add your required environment configurations:
```bash
   # Example configurations (adjust based on your chosen LLM provider)
   OPENAI_API_KEY=your_api_key_here
   ```

## 🖥️ Usage

1. **Place your documents:**
   Drop your files (.pdf, .csv) into the designated data/ source directory.

2. **Run the application:**
   Launch the main execution pipeline to ingest documents or initiate the interactive assistant:
   ```bash
   python main.py
   ```

3. **Run the Test Suite:**
   To verify the architecture integrity, execute::
   ```bash
   pytest
   ```
