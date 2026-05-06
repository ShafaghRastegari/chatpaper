# ChatPaper — Research Assistant Powered by RAG and AI Agents


ChatPaper is a local AI research assistant that lets you have a conversation with your academic papers. Upload PDFs, ask questions in plain English, and get cited, grounded answers and also find relevant paper! Built with production-grade RAG architecture, a LangGraph agent, and automated quality evaluation.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-7C3AED?style=flat&logo=data:image/svg+xml;base64,&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-F97316?style=flat&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace_BGE-FFD21E?style=flat&logo=huggingface&logoColor=black)
![OpenRouter](https://img.shields.io/badge/OpenRouter-6366F1?style=flat&logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS_Evaluation-10B981?style=flat&logoColor=white)

---

## What It Does

Most AI tools answer questions from their training data. ChatPaper answers from your documents. Every answer is grounded in specific pages of your uploaded papers, with citations included.

The system uses two retrieval strategies depending on the complexity of your question. Simple factual questions (authors, datasets, numbers) run a fast semantic search over the most relevant chunks. Complex questions (methodology, contributions, comparisons) send the entire paper content to the model for a thorough, structured answer.

---

## Features

**Core Capabilities**
- Upload one or many PDF papers and index them locally in 1–3 minutes
- Ask questions in plain English and receive cited answers
- Automatic mode switching. quick semantic search for factual questions, full-paper mode for complex ones
- Select which papers to chat with, query one paper, a subset, or all at once
- Multi-turn conversation with memory, follow-up questions work naturally

**Paper Discovery**
- Import papers directly from any arXiv URL
- Auto-suggest related papers from arXiv after indexing, based on extracted keywords
- Search arXiv's full database of 2M+ papers from within the app
- One-click download and immediate indexing of any arXiv paper

**Quality Evaluation**
- RAGAS evaluation after every answer
- Three metrics scored automatically: Faithfulness, Answer Relevancy, Context Precision
- Scores saved alongside chat history and reloaded when you revisit past conversations

**Chat Management**
- Conversations auto-saved after every message
- Full chat history with load and delete, restores the exact paper selection used
- Duplicate paper detection prevents re-indexing existing papers

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Streamlit UI                │
│         Upload · Chat · Find Papers          │
└──────────────────────┬──────────────────────┘
                       │
          ┌────────────▼────────────┐
          │    Query Router         │
          │  Simple? → RAG Search   │
          │  Complex? → Full Paper  │
          └────────────┬────────────┘
                       │
        ┌──────────────▼──────────────┐
        │       LlamaIndex RAG        │
        │  Chunk · Embed · Retrieve   │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │          ChromaDB           │
        │   Persistent Vector Store   │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │     Claude via OpenRouter   │
        │       Answer Generation     │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │       RAGAS Evaluation      │
        │  Faithfulness · Relevancy   │
        │     Context Precision       │
        └─────────────────────────────┘
```

---

## Project Structure

```
chatpaper/
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py        # PDF text extraction via PyMuPDF (page-by-page)
│   │   └── paper_fetcher.py     # arXiv API integration, keyword extraction, download
│   ├── rag/
│   │   └── pipeline.py          # LlamaIndex RAG pipeline + ChromaDB + full-paper mode
│   ├── agent/
│   │   ├── tools.py             # LangChain tools: search, compare, literature review
│   │   └── agent.py             # LangGraph ReAct agent with conversation memory
│   ├── evaluation/
│   │   └── ragas_eval.py        # RAGAS metrics: faithfulness, relevancy, precision
│   └── ui/
│       └── app.py               # Streamlit interface: chat, sidebar, history, find papers
├── chroma_db/                   # Auto-created: vector embeddings + metadata (persisted)
├── chats/                       # Auto-created: JSON conversation history
├── data/                        # Auto-created: downloaded arXiv PDFs
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## How It Works

**Indexing (runs once per paper)**

When you upload a PDF, PyMuPDF extracts the text page by page. LlamaIndex splits the text into 256-token chunks with 50-token overlaps, keeping context intact across chunk boundaries. Each chunk is converted into a 384-dimensional embedding vector using the BAAI/bge-small-en-v1.5 model running locally. The vectors and their metadata (filename, page number) are stored in ChromaDB on disk and persist across restarts.

**Querying (every question)**

The system first classifies your question. If it detects complexity keywords (explain, methodology, summarize, compare, etc.), it fetches all chunks from the selected paper and sends the complete content to the model. For simpler questions, it embeds the question into a vector, searches ChromaDB for the top-k most semantically similar chunks, and sends only those chunks as context. Either way, the model generates an answer grounded exclusively in the retrieved text.

**Evaluation (optional, per answer)**

When RAGAS evaluation is enabled, three additional LLM calls assess the answer quality. Faithfulness checks whether every claim in the answer is supported by the retrieved context. Answer Relevancy generates reverse questions from the answer and measures their cosine similarity to the original question. Context Precision judges whether the retrieved chunks were actually useful for answering the question.

---

## Getting Started

**Requirements:** Python 3.10+, an [OpenRouter](https://openrouter.ai) API key (models from ~$0.25/1M tokens)

```bash
# Clone and enter the project
git clone https://github.com/ShafaghRastegari/chatpaper.git
cd chatpaper

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Open .env and set OPENROUTER_API_KEY and OPENROUTER_MODEL

# Run the app
streamlit run src/ui/app.py
```

Open [http://localhost:8501](http://localhost:8501) and upload your first paper.

---

## Configuration

```env
# .env
# Fixes a protobuf conflict on Windows
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```


## Evaluation Results

Tested on a 17-page NLP research paper with a complex question set:

| Metric | Score | Interpretation |
|---|---|---|
| Faithfulness | **1.0** | Zero hallucinations — all claims supported by source text |
| Answer Relevancy | 0.59 | Answers address the question with some additional context |
| Context Precision | 0.0 | Known RAGAS limitation without ground-truth labels |

Faithfulness of 1.0 is the critical metric for a research assistant, it means the system never invents information that isn't in the paper.

---

## Design Decisions

**Why local embeddings?**
Running BAAI/bge-small-en-v1.5 locally means no second API key, no cost per embedding, and complete privacy, your paper content never leaves your machine during indexing.

**Why two retrieval modes?**
Top-k semantic search is fast and cheap for factual lookups. But questions about methodology or contributions require understanding the paper holistically, for those, sending all chunks to a 200k-context model is more accurate than hoping the right 5 chunks were retrieved.

**Why RAGAS with no ground truth?**
Most RAG systems ship with no evaluation at all. Even without human-labeled answers, faithfulness evaluation provides a meaningful signal about hallucination risk, which is the most important property for a research tool where accuracy is non-negotiable.

---

## Built With

| Layer | Technology | Role |
|---|---|---|
| Interface | Streamlit | Web UI with chat, sidebar, and tabs |
| Agent | LangGraph + LangChain | ReAct agent with tool calling |
| RAG | LlamaIndex | Document chunking, embedding, retrieval |
| Vector store | ChromaDB | Persistent local vector database |
| Embeddings | BAAI/bge-small-en-v1.5 | Free local text-to-vector model |
| LLM | Claude Haiku via OpenRouter | Answer generation and reasoning |
| PDF parsing | PyMuPDF | Fast, accurate PDF text extraction |
| Paper search | arXiv API | Free academic paper search and download |
| Evaluation | RAGAS | Automated RAG quality measurement |

---
---
 
## Deployment
 
ChatPaper is designed to run both locally and in production with no code changes. For deployment, the app connects to **Chroma Cloud** for persistent vector storage and **HuggingFace Hub** as a private dataset repository for chat history, replacing the local `chroma_db/` and `chats/` folders that are used during development.