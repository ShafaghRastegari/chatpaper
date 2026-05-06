from typing import Optional
from langchain_core.tools import tool

# We store the RAG pipeline as a module-level variable.
# This is a simple way to share the pipeline instance with tools
# without passing it through every function call.
# The set_rag_pipeline() function sets it at startup.
_rag_pipeline = None


def set_rag_pipeline(pipeline) -> None:
    global _rag_pipeline
    _rag_pipeline = pipeline
    print("RAG pipeline registered with agent tools")


# ─────────────────────────────────────────────────────────────
# TOOL 1: search_papers
# ─────────────────────────────────────────────────────────────

@tool
def search_papers(question: str) -> str:
    """
    Search through the uploaded research papers to answer a specific question.

    Use this tool when the user asks about:
    - Content, findings, or conclusions from the papers
    - Methods, datasets, or experiments described in the papers
    - Definitions or explanations that should be in the papers
    - Specific details like numbers, results, or quotes from papers

    The tool searches semantically — it understands meaning, not just keywords.

    Args:
        question: The specific question to answer from the papers

    Returns:
        An answer with source citations (paper name + page number)
    """
    if _rag_pipeline is None:
        return "Error: No papers indexed yet. Please upload and index PDFs first."

    try:
        result = _rag_pipeline.query(question)

        answer = result["answer"]
        sources = result["sources"]

        if sources:
            answer += "\n\n **Sources:**\n"
            for i, src in enumerate(sources, 1):
                score_str = f" (relevance: {src['relevance_score']})" if src["relevance_score"] else ""
                answer += f"  {i}. **{src['file_name']}** — Page {src['page_number']}{score_str}\n"

        return answer

    except Exception as e:
        return f"Search error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# TOOL 2: compare_papers
# ─────────────────────────────────────────────────────────────

@tool
def compare_papers(aspect_to_compare: str) -> str:
    """
    Compare how different papers approach or discuss a specific topic or aspect.

    Use this tool when the user asks to:
    - Compare or contrast papers on a topic
    - Find similarities or differences between papers
    - Understand different perspectives across papers
    - See which papers agree or disagree on something

    Args:
        aspect_to_compare: The specific aspect, topic, or concept to compare
                           across papers (e.g. "methodology", "results on dataset X")

    Returns:
        A comparative analysis citing multiple papers
    """
    if _rag_pipeline is None:
        return "Error: No papers indexed yet."

    # We reformulate the query to explicitly ask for comparison.
    comparison_prompt = (
        f"Compare and contrast what different papers say about: {aspect_to_compare}. "
        f"For each paper, describe its position or approach. "
        f"Then highlight key similarities and differences."
    )

    try:
        result = _rag_pipeline.query(comparison_prompt)
        return result["answer"]
    except Exception as e:
        return f"Comparison error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# TOOL 3: generate_literature_review
# ─────────────────────────────────────────────────────────────

@tool
def generate_literature_review(topic: str) -> str:
    """
    Generate a structured academic literature review on a topic
    based on the uploaded papers.

    Use this tool when the user asks to:
    - Write a literature review or summary section
    - Summarize the state of research on a topic
    - Get an overview of what all papers say about something
    - Generate academic-style text from the papers

    Args:
        topic: The research topic to write the literature review about

    Returns:
        A structured literature review paragraph in academic style
    """
    if _rag_pipeline is None:
        return "Error: No papers indexed yet."

    # This prompt guides Claude to write in academic literature review style
    review_prompt = f"""
    Write a structured academic literature review on the topic: "{topic}"
    
    Your review should:
    1. Introduce the topic and its importance
    2. Summarize key findings from the papers on this topic
    3. Identify different methodological approaches used
    4. Note areas of agreement and disagreement between papers
    5. Identify research gaps or limitations mentioned
    
    Use formal academic language. Reference specific papers when possible.
    Organize the content logically with clear flow between ideas.
    """

    try:
        result = _rag_pipeline.query(review_prompt)

        # Append sources at the end of the review
        review = result["answer"]
        sources = result["sources"]

        if sources:
            unique_papers = list({src["file_name"] for src in sources})
            review += f"\n\n*Based on: {', '.join(unique_papers)}*"

        return review

    except Exception as e:
        return f"Literature review error: {str(e)}"