# src/ui/app.py
import sys
import re
import os
import json
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

load_dotenv()

import streamlit as st
from src.rag.pipeline import RAGPipeline
from src.storage.hf_storage import (
    ensure_dataset_repo, save_chat, load_all_chats,
    delete_chat as hf_delete_chat,
    save_related_papers as hf_save_related_papers,
    load_related_papers as hf_load_related_papers,
)
from src.agent.tools import set_rag_pipeline
from src.agent.agent import ChatPaperAgent
from src.ingestion.pdf_loader import load_papers_from_folder
from src.ingestion.paper_fetcher import search_arxiv, find_related_papers, download_paper, download_from_arxiv_url
from src.evaluation.ragas_eval import evaluate_answer, get_score_emoji, format_score_bar

st.set_page_config(
    page_title="ChatPaper",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────
# Storage is handled by HuggingFace Hub (persistent across restarts)

# ── Related Papers Persistence ────────────────────────────────

def save_related_papers():
    try:
        hf_save_related_papers(st.session_state.related_papers)
    except Exception as e:
        print("Could not save related papers: " + str(e))

def load_related_papers():
    return hf_load_related_papers()

# ── Chat Storage ──────────────────────────────────────────────

def save_current_chat():
    if not st.session_state.chat_history:
        return
    session_id = st.session_state.session_id
    first_msg = st.session_state.chat_history[0]["content"]
    question_title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
    papers = st.session_state.selected_papers
    if papers:
        paper_short = Path(papers[0]).stem[:30]
        if len(papers) > 1:
            paper_short += " +" + str(len(papers) - 1) + " more"
        title = "[" + paper_short + "] " + question_title
    else:
        title = question_title
    chat_data = {
        "session_id": session_id,
        "title": title,
        "timestamp": st.session_state.session_timestamp,
        "papers": papers,
        "messages": st.session_state.chat_history,
    }
    save_chat(chat_data)

def delete_chat(session_id):
    hf_delete_chat(session_id)

def load_chat_session(chat_data):
    st.session_state.session_id = chat_data["session_id"]
    st.session_state.session_timestamp = chat_data["timestamp"]
    st.session_state.chat_history = chat_data["messages"]
    st.session_state.just_loaded_chat = True
    saved_papers = chat_data.get("papers", [])
    available = st.session_state.indexed_paper_names
    restored = [p for p in saved_papers if p in available]
    st.session_state.selected_papers = restored
    st.session_state["pending_checkbox_update"] = restored
    missing = [p for p in saved_papers if p not in available]
    if missing:
        st.warning(
            "⚠️ Some papers from this chat are no longer indexed:\n"
            + "\n".join("- " + m for m in missing)
        )

# ── ChromaDB Helper ───────────────────────────────────────────

def get_paper_names_from_chroma(pipeline):
    try:
        results = pipeline.chroma_collection.get(include=["metadatas"])
        names = list({
            m["file_name"]
            for m in results["metadatas"]
            if m and "file_name" in m
        })
        return sorted(names)
    except Exception:
        return []

# ── Session State ─────────────────────────────────────────────

def init_session_state():
    defaults = {
        "pipeline": None,
        "agent": None,
        "chat_history": [],
        "papers_indexed": False,
        "indexed_paper_names": [],
        "selected_papers": [],
        "related_papers": {},
        "search_results": [],
        "download_folder": "./data/downloaded_papers",
        "session_id": str(uuid.uuid4()),
        "session_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "show_history": False,
        "just_loaded_chat": False,
        "pending_checkbox_update": None,
        "ragas_enabled": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ── Initialization ────────────────────────────────────────────

def initialize_app():
    if st.session_state.pipeline is None:
        with st.spinner("🔧 Initializing pipeline..."):
            if os.getenv('HF_TOKEN'):
                ensure_dataset_repo()
            pipeline = RAGPipeline()
            if pipeline.load_existing_index():
                st.session_state.papers_indexed = True
                st.session_state.indexed_paper_names = get_paper_names_from_chroma(pipeline)
                st.session_state.selected_papers = list(st.session_state.indexed_paper_names)
                st.session_state.related_papers = load_related_papers()
            set_rag_pipeline(pipeline)
            st.session_state.pipeline = pipeline
    if st.session_state.agent is None:
        st.session_state.agent = ChatPaperAgent()

# ── Sidebar ───────────────────────────────────────────────────

def render_sidebar():
    # Apply pending checkbox updates BEFORE widgets are instantiated
    if st.session_state.get("pending_checkbox_update") is not None:
        restored = st.session_state["pending_checkbox_update"]
        for name in st.session_state.indexed_paper_names:
            st.session_state["chk_" + name] = name in restored
        st.session_state["pending_checkbox_update"] = None

    with st.sidebar:
        st.title("📚 ChatPaper")
        st.caption("AI-Powered Research Assistant")
        st.divider()

        # ── Upload ──────────────────────────────────────────
        st.subheader("📄 Upload Research Papers")
        uploaded_files = st.file_uploader(
            label="Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            existing = st.session_state.indexed_paper_names
            duplicates = [f.name for f in uploaded_files if f.name in existing]
            new_files = [f for f in uploaded_files if f.name not in existing]
            if duplicates:
                st.warning("⚠️ Already indexed:\n" + "\n".join("- " + d for d in duplicates))
            if new_files:
                st.caption("New: " + ", ".join(f.name for f in new_files))
                if st.button("🔄 Index Papers", type="primary", use_container_width=True):
                    handle_indexing(new_files)
            elif duplicates and not new_files:
                st.info("All papers already indexed.")

        # ── arXiv URL Import ─────────────────────────────────
        st.divider()
        st.subheader("🔗 Import from arXiv URL")
        arxiv_url_input = st.text_input(
            label="arXiv URL",
            placeholder="https://arxiv.org/abs/2305.12345",
            label_visibility="collapsed",
            key="arxiv_url_input",
        )
        if st.button("⬇️ Download & Index", key="arxiv_url_btn", use_container_width=True):
            if arxiv_url_input.strip():
                handle_arxiv_url_import(arxiv_url_input.strip())
            else:
                st.warning("Please enter an arXiv URL first.")

        # ── Status & Paper Selector ─────────────────────────
        st.divider()
        st.subheader("📊 Status")
        if st.session_state.papers_indexed:
            paper_count = len(st.session_state.indexed_paper_names)
            st.success("" + str(paper_count) + " paper(s) indexed")
            st.caption("🗂️ Select papers to chat with:")
            all_names = st.session_state.indexed_paper_names
            col_all, col_none = st.columns(2)
            with col_all:
                if st.button("All", use_container_width=True):
                    st.session_state.selected_papers = list(all_names)
                    st.session_state.chat_history = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.rerun()
            with col_none:
                if st.button("None", use_container_width=True):
                    st.session_state.selected_papers = []
                    st.rerun()
            newly_selected = []
            for name in all_names:
                checked = name in st.session_state.selected_papers
                if st.checkbox(label=name, value=checked, key="chk_" + name):
                    newly_selected.append(name)
            if set(newly_selected) != set(st.session_state.selected_papers):
                if st.session_state.just_loaded_chat:
                    st.session_state.selected_papers = newly_selected
                    st.session_state.just_loaded_chat = False
                else:
                    st.session_state.selected_papers = newly_selected
                    st.session_state.chat_history = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    if st.session_state.agent:
                        st.session_state.agent.reset()
                st.rerun()
            n = len(st.session_state.selected_papers)
            total = len(all_names)
            if n == 0:
                st.error("⚠️ No papers selected.")
            elif n == total:
                st.caption("💬 Chatting with all " + str(total) + " papers")
            else:
                st.caption("💬 Chatting with " + str(n) + " of " + str(total) + " papers")
        else:
            st.info("📂 No papers indexed yet")

        # ── Chat Controls ────────────────────────────────────
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                if st.session_state.agent:
                    st.session_state.agent.reset()
                st.rerun()
        with col2:
            if st.button("💾 Save Chat", use_container_width=True):
                if st.session_state.chat_history:
                    save_current_chat()
                    st.success("Saved!")
                else:
                    st.warning("Nothing to save.")

        # ── RAGAS Toggle ─────────────────────────────────────
        st.divider()
        st.session_state.ragas_enabled = st.toggle(
            "📊 Enable RAGAS Evaluation",
            value=st.session_state.ragas_enabled,
            help="Score each answer for faithfulness, relevancy, and context precision."
        )
        if st.session_state.ragas_enabled:
            st.caption("Each answer will be scored after generation.")

        # ── Chat History ─────────────────────────────────────
        st.divider()
        st.subheader("🕓 Chat History")
        all_chats = load_all_chats()
        if not all_chats:
            st.caption("No saved chats yet.")
        else:
            for chat in all_chats:
                with st.container(border=True):
                    st.caption(chat.get("timestamp", ""))
                    st.markdown("**" + chat["title"] + "**")
                    papers = chat.get("papers", [])
                    if papers:
                        st.caption("📄 " + ", ".join(Path(p).stem[:20] for p in papers[:2]))
                    col_load, col_del = st.columns(2)
                    with col_load:
                        if st.button("📂 Load", key="load_" + chat["session_id"], use_container_width=True):
                            load_chat_session(chat)
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key="del_" + chat["session_id"], use_container_width=True):
                            delete_chat(chat["session_id"])
                            st.rerun()

        # ── Tips ─────────────────────────────────────────────
        st.divider()
        st.subheader("💡 Try asking:")
        st.markdown("""
        - *What is the main contribution?*
        - *Explain the methodology*
        - *What are the limitations?*
        - *Summarize the findings*
        - *Which paper performs best?*
        """)

# ── arXiv URL Import Handler ──────────────────────────────────

def handle_arxiv_url_import(url: str):
    folder = st.session_state.download_folder
    with st.spinner("📄 Fetching paper from arXiv..."):
        try:
            pdf_path, metadata = download_from_arxiv_url(url, folder)
            st.success("✅ Downloaded: " + metadata["title"][:60])
        except ValueError as e:
            st.error("❌ Invalid URL: " + str(e))
            return
        except Exception as e:
            st.error("❌ Download failed: " + str(e))
            return
    paper_name = Path(pdf_path).name
    if paper_name in st.session_state.indexed_paper_names:
        st.warning("⚠️ Already indexed: " + paper_name)
        return
    with st.spinner("🔄 Indexing paper..."):
        try:
            st.session_state.pipeline.index_papers(folder)
            set_rag_pipeline(st.session_state.pipeline)
            st.session_state.papers_indexed = True
            st.session_state.indexed_paper_names = get_paper_names_from_chroma(st.session_state.pipeline)
            if paper_name not in st.session_state.selected_papers:
                st.session_state.selected_papers.append(paper_name)
            st.success("✅ Indexed and ready to chat!")
        except Exception as e:
            st.error("❌ Indexing failed: " + str(e))
            return
    with st.spinner("🔍 Finding related papers..."):
        try:
            related = find_related_papers(
                paper_text=metadata.get("summary", ""),
                paper_title=metadata.get("title", ""),
                max_results=6,
            )
            st.session_state.related_papers[paper_name] = related
            save_related_papers()
        except Exception:
            pass
    st.rerun()

# ── Indexing ──────────────────────────────────────────────────

def handle_indexing(uploaded_files):
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            save_path = Path(tmp_dir) / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        with st.spinner("🔄 Indexing " + str(len(uploaded_files)) + " paper(s)..."):
            try:
                st.session_state.pipeline.index_papers(tmp_dir)
                set_rag_pipeline(st.session_state.pipeline)
                st.session_state.papers_indexed = True
                st.session_state.indexed_paper_names = get_paper_names_from_chroma(st.session_state.pipeline)
                for f in uploaded_files:
                    if f.name not in st.session_state.selected_papers:
                        st.session_state.selected_papers.append(f.name)
                st.success("✅ " + str(len(uploaded_files)) + " paper(s) indexed!")
            except Exception as e:
                st.error("❌ Indexing failed: " + str(e))
                return
        with st.spinner("🔍 Finding related papers..."):
            try:
                papers_data = load_papers_from_folder(tmp_dir)
                for paper_data in papers_data:
                    name = paper_data["metadata"]["file_name"]
                    title = paper_data["metadata"].get("title", "") or name
                    related = find_related_papers(paper_text=paper_data["text"][:5000], paper_title=title, max_results=6)
                    st.session_state.related_papers[name] = related
                save_related_papers()
            except Exception as e:
                st.warning("⚠️ Could not fetch related papers: " + str(e))
    st.rerun()

# ── Paper Card ────────────────────────────────────────────────

def render_paper_card(paper, key_prefix):
    with st.container(border=True):
        col_title, col_year = st.columns([5, 1])
        with col_title:
            st.markdown("**" + paper["title"] + "**")
        with col_year:
            st.caption(paper["published"])
        st.caption("👤 " + paper["authors"])
        st.markdown("_" + paper["summary"] + "_")
        col_view, col_dl = st.columns(2)
        with col_view:
            st.link_button("🔗 View on arXiv", paper["arxiv_url"], use_container_width=True)
        with col_dl:
            if st.button("⬇️ Download & Index", key=key_prefix + "_" + paper["id"], use_container_width=True):
                handle_download_and_index(paper)

def handle_download_and_index(paper):
    folder = st.session_state.download_folder
    filename = paper["id"] + "_" + paper["title"][:40].replace(" ", "_")
    filename = "".join(c for c in filename if c.isalnum() or c in "._-") + ".pdf"
    with st.spinner("⬇️ Downloading..."):
        try:
            pdf_path = download_paper(pdf_url=paper["pdf_url"], save_folder=folder, filename=filename)
        except Exception as e:
            st.error("❌ Download failed: " + str(e))
            return
    with st.spinner("🔄 Indexing..."):
        try:
            st.session_state.pipeline.index_papers(folder)
            set_rag_pipeline(st.session_state.pipeline)
            st.session_state.papers_indexed = True
            st.session_state.indexed_paper_names = get_paper_names_from_chroma(st.session_state.pipeline)
            paper_name = Path(pdf_path).name
            if paper_name not in st.session_state.selected_papers:
                st.session_state.selected_papers.append(paper_name)
            st.success("✅ Added and indexed!")
            st.rerun()
        except Exception as e:
            st.error("❌ Indexing failed: " + str(e))

# ── CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Chat input — responsive, never overlaps sidebar ── */
    .stChatInput {
        position: fixed;
        bottom: 0;
        right: 0;
        left: var(--sidebar-width, 0px);
        z-index: 999;
        padding: 0.75rem 1.5rem;
        background: transparent;
        border-top: none;
    }

    /* On wide screens where sidebar is visible, offset by sidebar width */
    @media (min-width: 768px) {
        .stChatInput {
            left: 21rem;
        }
    }

    /* Extra padding at bottom so last message is not hidden behind input */
    .main .block-container {
        padding-bottom: 160px !important;
    }

    /* Remove black background from chat messages area */
    .stChatMessage {
        background: transparent !important;
    }

    /* Fix active papers banner black background */
    .active-papers-banner {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: inherit !important;
    }
    .active-papers-banner span {
        color: #818cf8 !important;
    }

    /* Make page not scroll — only inner content scrolls */
    section.main > div {
        overflow-y: auto;
        height: calc(100vh - 80px);
    }

    /* ── Remove emoji from expander headers ── */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* ── Active papers banner ── */
    .active-papers-banner {
        background: #1a1d27;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.78rem;
        color: #9ca3af;
        margin-bottom: 12px;
    }
    .active-papers-banner span {
        color: #818cf8;
        font-weight: 600;
    }

    /* ── Mode badge on messages ── */
    .mode-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
        letter-spacing: 0.3px;
    }
    .mode-simple {
        background: #1a3a2a;
        color: #4ade80;
        border: 1px solid #166534;
    }
    .mode-complex {
        background: #1e1e35;
        color: #818cf8;
        border: 1px solid #3730a3;
    }

    /* ── RAGAS scores inline ── */
    .ragas-inline {
        margin-top: 10px;
        padding: 10px 14px;
        background: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
    }
    .ragas-inline-title {
        font-size: 0.7rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .ragas-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
    }
    .ragas-cell {
        text-align: center;
    }
    .ragas-cell-label {
        font-size: 0.68rem;
        color: #6b7280;
        margin-bottom: 2px;
    }
    .ragas-cell-value {
        font-size: 1.1rem;
        font-weight: 700;
    }
    .ragas-cell-bar {
        height: 3px;
        background: #262730;
        border-radius: 2px;
        margin-top: 3px;
        overflow: hidden;
    }
    .ragas-cell-fill {
        height: 100%;
        border-radius: 2px;
    }
    .score-green { color: #4ade80; }
    .score-orange { color: #fb923c; }
    .score-red { color: #f87171; }
    .fill-green { background: #4ade80; }
    .fill-orange { background: #fb923c; }
    .fill-red { background: #f87171; }

    /* ── Sidebar paper checkboxes — tighter ── */
    .stCheckbox label {
        font-size: 0.78rem !important;
    }

    /* ── Status badge ── */
    .status-green {
        background: #1a3a2a;
        border: 1px solid #166534;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 0.78rem;
        color: #4ade80;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Chat Tab ──────────────────────────────────────────────────

def render_chat_tab():
    if not st.session_state.papers_indexed:
        st.markdown("### 👋 Welcome to ChatPaper!")
        st.info("Upload and index research papers using the sidebar to get started.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📖 Answer Questions**")
            st.caption("Precise answers from your papers with page citations")
        with col2:
            st.markdown("**⚖️ Compare Papers**")
            st.caption("Analyze differences in methodology and results")
        with col3:
            st.markdown("**📝 Literature Reviews**")
            st.caption("Auto-generate academic summaries")
        return

    if not st.session_state.selected_papers:
        st.warning("⚠️ No papers selected. Please select at least one paper from the sidebar.")
        return

    # Active papers banner
    paper_names_short = " · ".join(Path(p).stem[:25] for p in st.session_state.selected_papers[:3])
    if len(st.session_state.selected_papers) > 3:
        paper_names_short += " +" + str(len(st.session_state.selected_papers) - 3) + " more"
    st.markdown(
        '<div class="active-papers-banner">Chatting with <span>' + paper_names_short + '</span></div>',
        unsafe_allow_html=True
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Re-render RAGAS scores if they were saved with this message
            scores = message.get("ragas_scores")
            if scores:
                with st.expander("📊 Answer Quality Scores", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        score = scores["faithfulness"]
                        st.metric(label=get_score_emoji(score) + " Faithfulness", value=str(score))
                        st.caption(format_score_bar(score))
                    with col2:
                        score = scores["answer_relevancy"]
                        st.metric(label=get_score_emoji(score) + " Relevancy", value=str(score))
                        st.caption(format_score_bar(score))
                    with col3:
                        score = scores["context_precision"]
                        st.metric(label=get_score_emoji(score) + " Context Precision", value=str(score))
                        st.caption(format_score_bar(score))

    if user_input := st.chat_input("Ask anything about the selected paper(s)..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        response = ""
        ragas_scores = None
        contexts = []

        with st.chat_message("assistant"):
            with st.status("🤔 Researching papers...", expanded=True):
                try:
                    pipeline = st.session_state.pipeline
                    selected = st.session_state.selected_papers
                    is_complex = pipeline.is_complex_question(user_input)

                    if is_complex:
                        st.write("📖 Complex question — reading full paper...")
                        result = pipeline.query_full_paper(user_input, selected)
                    else:
                        st.write("🔍 Searching papers...")
                        result = pipeline.query(user_input)

                    response = result["answer"]
                    contexts = [src.get("excerpt", "") for src in result.get("sources", [])]

                    if result["sources"]:
                        seen = set()
                        unique_sources = []
                        for src in result["sources"]:
                            key = (src["file_name"], src["page_number"])
                            if key not in seen:
                                seen.add(key)
                                unique_sources.append(src)
                        response += "\n\n📚 **Sources:**\n"
                        for src in unique_sources[:3]:
                            response += "- **" + src["file_name"] + "** — Page " + str(src["page_number"]) + "\n"

                    st.write("✅ Done!")

                except Exception as e:
                    response = "⚠️ Something went wrong: " + str(e)
                    st.write("❌ Error occurred")

            if response:
                st.markdown(response)
            else:
                st.warning("No response returned. Try rephrasing your question.")

            # RAGAS evaluation — runs after answer is displayed
            if st.session_state.ragas_enabled and response and contexts:
                with st.spinner("📊 Evaluating answer quality..."):
                    ragas_scores = evaluate_answer(
                        question=user_input,
                        answer=response,
                        contexts=contexts,
                    )

            if ragas_scores:
                with st.expander("📊 Answer Quality Scores", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        score = ragas_scores["faithfulness"]
                        st.metric(
                            label=get_score_emoji(score) + " Faithfulness",
                            value=str(score),
                            help="Is the answer grounded in the retrieved text? High = no hallucination."
                        )
                        st.caption(format_score_bar(score))
                    with col2:
                        score = ragas_scores["answer_relevancy"]
                        st.metric(
                            label=get_score_emoji(score) + " Relevancy",
                            value=str(score),
                            help="Does the answer actually address the question?"
                        )
                        st.caption(format_score_bar(score))
                    with col3:
                        score = ragas_scores["context_precision"]
                        st.metric(
                            label=get_score_emoji(score) + " Context Precision",
                            value=str(score),
                            help="Were the right chunks retrieved from the paper?"
                        )
                        st.caption(format_score_bar(score))

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "ragas_scores": ragas_scores,
        })
        save_current_chat()

# ── Find Papers Tab ───────────────────────────────────────────

def fetch_related_papers_for_all():
    pipeline = st.session_state.pipeline
    all_names = st.session_state.indexed_paper_names
    st.info("🔍 Searching arXiv... this may take 10-30 seconds.")
    for i, name in enumerate(all_names):
        if name in st.session_state.related_papers:
            st.write("⏭️ Already fetched: " + name[:50])
            continue
        st.write("🔍 Searching for: **" + name[:50] + "**")
        try:
            results = pipeline.chroma_collection.get(
                where={"file_name": {"$eq": name}},
                include=["documents", "metadatas"]
            )
            if not results["documents"]:
                st.write("⚠️ No chunks found for: " + name)
                continue
            text_sample = " ".join(results["documents"][:3])[:5000]
            title = name.replace(".pdf", "")
            related = find_related_papers(paper_text=text_sample, paper_title=title, max_results=6)
            st.session_state.related_papers[name] = related
            st.write("✅ Found " + str(len(related)) + " related papers")
        except Exception as e:
            st.write("❌ Error for " + name[:40] + ": " + str(e))
            st.session_state.related_papers[name] = []
    save_related_papers()
    st.success("✅ Done!")
    st.rerun()

def render_find_papers_tab():
    st.subheader("🔗 Related Papers — Based on Your Uploaded Papers")
    if not st.session_state.related_papers:
        st.info("📂 Upload and index a paper — related papers appear here automatically.")
        if st.session_state.papers_indexed:
            if st.button("🔍 Find Related Papers Now", type="primary"):
                fetch_related_papers_for_all()
    else:
        for source_paper, related_list in st.session_state.related_papers.items():
            with st.expander("📄 Related to: **" + source_paper + "**", expanded=True):
                if not related_list:
                    st.caption("No related papers found.")
                    continue
                cols = st.columns(2)
                for i, paper in enumerate(related_list):
                    with cols[i % 2]:
                        safe_source = re.sub(r"[^a-zA-Z0-9]", "", source_paper[:15])
                        render_paper_card(paper, key_prefix="rel_" + safe_source + "_" + str(i))

    st.divider()
    st.subheader("🔍 Search arXiv for Papers")
    st.caption("Search over 2 million free papers — no API key needed.")
    search_col, btn_col = st.columns([4, 1])
    with search_col:
        query = st.text_input(
            label="query",
            placeholder="e.g. transformer attention, diffusion models",
            label_visibility="collapsed"
        )
    with btn_col:
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    if search_clicked and query.strip():
        with st.spinner("🔍 Searching arXiv..."):
            results = search_arxiv(query.strip(), max_results=8)
            st.session_state.search_results = results
        if not results:
            st.warning("No results found.")
    if st.session_state.search_results:
        st.markdown("**" + str(len(st.session_state.search_results)) + " results:**")
        cols = st.columns(2)
        for i, paper in enumerate(st.session_state.search_results):
            with cols[i % 2]:
                render_paper_card(paper, key_prefix="srch_" + str(i))

# ── Main ──────────────────────────────────────────────────────

def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("❌ OPENROUTER_API_KEY not found!")
        st.markdown("Add it to your `.env` file. Get your key at https://openrouter.ai/keys")
        st.stop()

    init_session_state()
    initialize_app()
    render_sidebar()

    st.title("🔬 ChatPaper Research Assistant")
    tab_chat, tab_find = st.tabs(["💬 Chat with Papers", "🔍 Find Papers"])
    with tab_chat:
        render_chat_tab()
    with tab_find:
        render_find_papers_tab()

if __name__ == "__main__":
    main()