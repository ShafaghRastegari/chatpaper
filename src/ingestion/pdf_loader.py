import fitz  # This is PyMuPDF — the package is called pymupdf but imports as fitz
from pathlib import Path
from typing import List, Dict, Any


def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract all text and metadata from a single PDF file.

    This function opens a PDF and reads it page by page.
    We process per-page rather than extracting everything at once
    because it lets us later tell the user *which page* an answer came from.

    Args:
        pdf_path (str): Full file path to the PDF

    Returns:
        dict with three keys:
          - "text"     → the entire paper as one string
          - "metadata" → title, author, page count, filename
          - "pages"    → list of {page_number, text} per page
    """
    doc = fitz.open(pdf_path)

    pages = []       
    full_text = ""  

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")
        pages.append({
            "page_number": page_num + 1,
            "text": page_text
        })

        full_text += page_text + "\n"


    metadata = doc.metadata
    metadata["file_name"] = Path(pdf_path).name          
    metadata["total_pages"] = len(doc)
    metadata["file_path"] = str(pdf_path)

    doc.close()

    return {
        "text": full_text,
        "metadata": metadata,
        "pages": pages
    }


def load_papers_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load every PDF in a folder and return their extracted content.

    This is the main entry point called by the RAG pipeline.
    It scans the folder, processes each PDF, and returns a list
    ready to be embedded and stored.

    Args:
        folder_path (str): Path to a directory containing PDF files

    Returns:
        List of paper dicts (each from extract_text_from_pdf)

    Raises:
        FileNotFoundError: If the folder doesn't exist
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # glob("*.pdf") finds all files ending in .pdf (case-sensitive on Linux)
    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []

    papers = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        try:
            paper_data = extract_text_from_pdf(str(pdf_file))
            pages = paper_data["metadata"]["total_pages"]
            print(f"{pages} pages extracted")
            papers.append(paper_data)
        except Exception as e:
            print(f"Skipping {pdf_file.name}: {e}")

    print(f"\nLoaded {len(papers)} / {len(pdf_files)} papers")
    return papers
