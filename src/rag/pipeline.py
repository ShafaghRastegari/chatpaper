import os
from typing import List, Optional

import chromadb
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding


from src.ingestion.pdf_loader import load_papers_from_folder

COLLECTION_NAME = "research_papers"
_models_initialized = False


def setup_models() -> None:
    global _models_initialized
    if _models_initialized:
        return
    print("Setting up models...")
    Settings.llm = OpenAILike(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        max_tokens=2048,
        is_chat_model=True,
        context_window=32000,
    )
    Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
    )
    _models_initialized = True
    print("Models configured")


def build_documents(papers: List[dict]) -> List[Document]:
    documents = []
    for paper in papers:
        file_name = paper["metadata"]["file_name"]
        title = paper["metadata"].get("title") or file_name
        for page_data in paper["pages"]:
            text = page_data["text"].strip()
            if len(text) < 100:
                continue
            doc = Document(
                text=text,
                metadata={
                    "file_name": file_name,
                    "title": title,
                    "author": paper["metadata"].get("author", "Unknown"),
                    "page_number": page_data["page_number"],
                    "total_pages": paper["metadata"]["total_pages"],
                }
            )
            documents.append(doc)
    print(f"Created {len(documents)} document pages from {len(papers)} papers")
    return documents


def create_chroma_client():
    api_key = os.getenv("CHROMA_API_KEY")
    if api_key:
        print("Connecting to Chroma Cloud...")
        kwargs = dict(
            api_key=api_key,
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
            cloud_host=os.getenv("CHROMA_HOST"),
            cloud_port=443,
        )
        return chromadb.CloudClient(**kwargs), True
    else:
        print("Using local ChromaDB...")
        return chromadb.PersistentClient(path="./chroma_db"), False


class RAGPipeline:

    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None

        setup_models()

        self.chroma_client, self.is_cloud = create_chroma_client()

        if self.is_cloud:
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
            )
        else:
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        count = self.chroma_collection.count()
        print(f"ChromaDB ready — {count} chunks stored")

    def get_indexed_paper_names_set(self) -> set:
        """Get set of paper names already indexed — used for duplicate check."""
        results = self.chroma_collection.get(
            include=["metadatas"],
            limit=100,
        )
        names = set()
        offset = 0
        while True:
            results = self.chroma_collection.get(
                include=["metadatas"],
                limit=100,
                offset=offset,
            )
            if not results["metadatas"]:
                break
            for m in results["metadatas"]:
                if m and "file_name" in m:
                    names.add(m["file_name"])
            if len(results["metadatas"]) < 100:
                break
            offset += 100
        return names

    def index_papers(self, papers_folder: str) -> None:
        print(f"Indexing papers from: {papers_folder}")

        papers = load_papers_from_folder(papers_folder)
        if not papers:
            raise ValueError("No PDF files found.")

        # Check which papers are already indexed, skip duplicates
        already_indexed = self.get_indexed_paper_names_set()
        new_papers = [
            p for p in papers
            if p["metadata"]["file_name"] not in already_indexed
        ]

        if not new_papers:
            print("All papers already indexed — nothing to do.")
            self._rebuild_index_if_needed()
            return

        skipped = len(papers) - len(new_papers)
        if skipped > 0:
            print(f"Skipping {skipped} already-indexed paper(s)")

        documents = build_documents(new_papers)
        splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("Embedding chunks — this takes 1-3 minutes on first run...")
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=True,
        )

        print(f"Indexing complete! {self.chroma_collection.count()} chunks stored.")
        self._build_query_engine()

    def _rebuild_index_if_needed(self) -> None:
        """Rebuild query engine from existing index if not already built."""
        if self.query_engine is None:
            self.load_existing_index()

    def load_existing_index(self) -> bool:
        count = self.chroma_collection.count()
        if count == 0:
            print("ChromaDB is empty. Please index papers first.")
            return False

        print(f"Loading existing index ({count} chunks)...")
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
        self._build_query_engine()
        print("Index loaded — ready to answer questions!")
        return True

    def _build_query_engine(self) -> None:
        if self.index is None:
            raise RuntimeError("Index not initialized.")
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )

    def query(self, question: str) -> dict:
        if self.query_engine is None:
            raise RuntimeError("Query engine not ready.")

        response = self.query_engine.query(question)

        sources = []
        seen = set()
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                key = (
                    node.metadata.get("file_name", "Unknown"),
                    node.metadata.get("page_number", "?")
                )
                if key in seen:
                    continue
                seen.add(key)
                sources.append({
                    "file_name": node.metadata.get("file_name", "Unknown"),
                    "page_number": node.metadata.get("page_number", "?"),
                    "title": node.metadata.get("title", "Unknown"),
                    "relevance_score": round(node.score, 3) if node.score else None,
                    "excerpt": node.text[:200].replace("\n", " ") + "...",
                })
                if len(sources) >= 3:
                    break

        return {"answer": str(response), "sources": sources}

    def query_full_paper(self, question: str, paper_names: list) -> dict:
        all_docs = []
        all_metas = []

        if len(paper_names) == 1:
            where_filter = {"file_name": {"$eq": paper_names[0]}}
        else:
            where_filter = {"$or": [{"file_name": {"$eq": n}} for n in paper_names]}

        # Fetch in batches of 100 to avoid timeout
        offset = 0
        batch_size = 100
        while True:
            results = self.chroma_collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            if not results["documents"]:
                break
            all_docs.extend(results["documents"])
            all_metas.extend(results["metadatas"])
            if len(results["documents"]) < batch_size:
                break
            offset += batch_size

        if not all_docs:
            return {"answer": "No content found for the selected papers.", "sources": []}

        chunks = list(zip(all_docs, all_metas))
        chunks.sort(key=lambda x: (x[1].get("file_name", ""), x[1].get("page_number", 0)))

        context_parts = []
        seen_pages = set()
        sources = []

        for doc, meta in chunks:
            page = meta.get("page_number", "?")
            fname = meta.get("file_name", "Unknown")
            page_key = (fname, page)
            context_parts.append("[" + fname + " — Page " + str(page) + "]\n" + doc)
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                if len(sources) < 3:
                    sources.append({
                        "file_name": fname,
                        "page_number": page,
                        "title": meta.get("title", fname),
                        "relevance_score": None,
                        "excerpt": doc[:200].replace("\n", " ") + "...",
                    })

        full_context = "\n\n".join(context_parts)
        prompt = (
            "You are analyzing a research paper. Below is the COMPLETE content.\n\n"
            + full_context
            + "\n\n---\n\nAnswer this question thoroughly:\n"
            + question
            + "\n\nProvide a detailed, well-structured answer with page references."
        )

        response = Settings.llm.complete(prompt)
        return {"answer": str(response), "sources": sources}

    @staticmethod
    def is_complex_question(question: str) -> bool:
        complex_keywords = [
            "explain", "methodology", "summarize", "summary", "overview",
            "how does", "how do", "describe", "elaborate", "detail",
            "approach", "framework", "architecture", "algorithm", "process",
            "contribution", "contributions", "findings", "conclusion",
            "compare", "difference", "similar", "versus", "vs",
            "literature review", "related work", "background",
            "entire", "whole", "full", "complete", "overall",
            "what is the", "how is the", "walk me through",
        ]
        return any(kw in question.lower() for kw in complex_keywords)

    def get_paper_names(self) -> list:
        all_names = set()
        offset = 0
        batch_size = 100
        while True:
            results = self.chroma_collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            if not results["metadatas"]:
                break
            for m in results["metadatas"]:
                if m and "file_name" in m:
                    all_names.add(m["file_name"])
            if len(results["metadatas"]) < batch_size:
                break
            offset += batch_size
        return sorted(list(all_names))

    def get_chunk_count(self) -> int:
        return self.chroma_collection.count()