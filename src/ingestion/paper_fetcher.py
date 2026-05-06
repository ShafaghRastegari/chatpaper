import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from pathlib import Path


# arXiv XML namespace
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


# ─────────────────────────────────────────────────────────────
# Search papers from arXiv
# ─────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers matching a query string.

    Sends a request to the arXiv API and parses the XML response.
    Returns a list of paper dicts with metadata and direct PDF links.

    Args:
        query:       Search query (e.g. "attention transformer NLP")
        max_results: How many papers to return (default 8)

    Returns:
        List of dicts, each with:
          - id          : arXiv ID (e.g. "2305.12345")
          - title       : Paper title
          - authors     : Comma-separated author names
          - summary     : Abstract (truncated to 300 chars for display)
          - published   : Publication year
          - pdf_url     : Direct link to download the PDF (always free)
          - arxiv_url   : Link to the arXiv page
    """
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",        
        "sortOrder": "descending",
    })

    url = f"https://export.arxiv.org/api/query?{params}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ChatPaper-Research-Assistant/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                xml_data = response.read().decode("utf-8")
            break  
        except Exception as e:
            print(f"arXiv API error (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                wait = (attempt + 1) * 3
                print(f"   Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return []

    return _parse_arxiv_xml(xml_data)


def _parse_arxiv_xml(xml_data: str) -> List[Dict[str, Any]]:
    """
    Parse arXiv API XML response into a list of paper dicts.

    arXiv returns Atom feed XML. Each <entry> tag is one paper.
    We extract the fields we need and build clean dicts.

    Args:
        xml_data: Raw XML string from arXiv API

    Returns:
        List of paper dicts
    """
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        return []

    papers = []

    for entry in root.findall("atom:entry", ARXIV_NS):

        id_tag = entry.find("atom:id", ARXIV_NS)
        if id_tag is None:
            continue
        arxiv_id = id_tag.text.split("/abs/")[-1].split("v")[0]

        title_tag = entry.find("atom:title", ARXIV_NS)
        title = " ".join(title_tag.text.split()) if title_tag is not None else "Unknown Title"

        authors = []
        for author in entry.findall("atom:author", ARXIV_NS):
            name_tag = author.find("atom:name", ARXIV_NS)
            if name_tag is not None:
                authors.append(name_tag.text)
        if len(authors) > 3:
            authors_str = ", ".join(authors[:3]) + " et al."
        else:
            authors_str = ", ".join(authors)

        summary_tag = entry.find("atom:summary", ARXIV_NS)
        summary = " ".join(summary_tag.text.split()) if summary_tag is not None else ""
        summary_short = summary[:300] + "..." if len(summary) > 300 else summary

        published_tag = entry.find("atom:published", ARXIV_NS)
        published = published_tag.text[:4] if published_tag is not None else "?"

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors_str,
            "summary": summary_short,
            "published": published,
            "pdf_url": pdf_url,
            "arxiv_url": arxiv_url,
        })

    return papers


# ─────────────────────────────────────────────────────────────
# Extract keywords from an uploaded paper, suggest related papers
# ─────────────────────────────────────────────────────────────

def extract_keywords_from_text(text: str, max_keywords: int = 6) -> List[str]:
    """
    Extract the most important keywords from a paper's text.

    This is a lightweight approach that doesn't need an LLM or NLTK:
      1. Look for an explicit "Keywords:" section (many papers have this)
      2. Fall back to finding frequent meaningful words in the abstract

    Args:
        text:         Full text of the paper
        max_keywords: How many keywords to return

    Returns:
        List of keyword strings
    """
    # Strategy 1: Look for an explicit "Keywords" section
    # Many academic papers have "Keywords: transformer, attention, NLP"
    keyword_pattern = re.search(
        r"[Kk]eywords?\s*[:\-]\s*(.+?)(?:\n\n|\n[A-Z]|\Z)",
        text[:3000],  # Only look in first 3000 chars (usually in abstract section)
        re.DOTALL
    )

    if keyword_pattern:
        raw_keywords = keyword_pattern.group(1)
        keywords = re.split(r"[,;\n]", raw_keywords)
        keywords = [k.strip() for k in keywords if len(k.strip()) > 3]
        if keywords:
            return keywords[:max_keywords]

    # Strategy 2: Extract frequent meaningful words from the abstract
    # The abstract is usually in the first 2000 characters
    abstract_text = text[:2000].lower()

    # Remove common academic stop words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "this", "that", "these", "those", "we", "our", "their", "which",
        "also", "can", "its", "it", "as", "not", "than", "more", "such",
        "paper", "propose", "proposed", "show", "shows", "model", "based",
        "using", "use", "used", "method", "approach", "results", "result",
    }

    # Find all words of 4+ characters
    words = re.findall(r"\b[a-z]{4,}\b", abstract_text)

    # Count word frequency
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, freq in sorted_words[:max_keywords] if freq > 1]

    return keywords if keywords else ["machine learning", "deep learning"]


def find_related_papers(paper_text: str, paper_title: str = "", max_results: int = 6) -> List[Dict[str, Any]]:
    """
    Automatically find papers related to an uploaded paper.

    This is called automatically after indexing — no user input needed.
    It extracts keywords from the paper and searches arXiv for them.

    Args:
        paper_text:  Full text of the uploaded paper
        paper_title: Title of the paper (used to improve the search)
        max_results: Number of related papers to return

    Returns:
        List of related paper dicts (same format as search_arxiv)
    """
    print(f"Finding related papers for: {paper_title or 'uploaded paper'}")

    # Extract keywords from the paper text
    keywords = extract_keywords_from_text(paper_text)
    print(f"   Keywords found: {', '.join(keywords)}")

    # Build a search query from the keywords
    title_words = " ".join(paper_title.split()[:5]) if paper_title else ""
    keyword_query = " ".join(keywords[:4])
    query = f"{title_words} {keyword_query}".strip()
    time.sleep(2)

    papers = search_arxiv(query, max_results=max_results)
    print(f"   Found {len(papers)} related papers")
    return papers



# ─────────────────────────────────────────────────────────────
# Download directly from arXiv URL
# ─────────────────────────────────────────────────────────────

def parse_arxiv_url(url: str) -> str:
    """
    Extract arXiv ID from any arXiv URL format.

    Handles all common arXiv URL formats:
      - https://arxiv.org/abs/2305.12345
      - https://arxiv.org/pdf/2305.12345
      - https://arxiv.org/abs/2305.12345v2
      - 2305.12345 (just the ID)

    Args:
        url: arXiv URL or ID string

    Returns:
        Clean arXiv ID (e.g. "2305.12345")

    Raises:
        ValueError: If the URL doesn't look like an arXiv link
    """
    url = url.strip()

    # If it's just an ID like "2305.12345"
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", url):
        return url.split("v")[0]

    match = re.search(r"arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})", url)
    if match:
        return match.group(2)

    raise ValueError(
        "Could not extract arXiv ID from: " + url +
        "\nExpected format: https://arxiv.org/abs/2305.12345"
    )


def fetch_paper_metadata(arxiv_id: str) -> dict:
    """
    Fetch title and authors for a paper by arXiv ID.

    Args:
        arxiv_id: Clean arXiv ID (e.g. "2305.12345")

    Returns:
        dict with title, authors, summary, pdf_url, arxiv_url
    """
    params = urllib.parse.urlencode({
        "id_list": arxiv_id,
        "max_results": 1,
    })
    url = "https://export.arxiv.org/api/query?" + params

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ChatPaper-Research-Assistant/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read().decode("utf-8")
        papers = _parse_arxiv_xml(xml_data)
        if papers:
            return papers[0]
    except Exception as e:
        print("Could not fetch metadata: " + str(e))

    # Fallback if API fails
    return {
        "id": arxiv_id,
        "title": arxiv_id,
        "authors": "Unknown",
        "summary": "",
        "published": "?",
        "pdf_url": "https://arxiv.org/pdf/" + arxiv_id + ".pdf",
        "arxiv_url": "https://arxiv.org/abs/" + arxiv_id,
    }


def download_from_arxiv_url(arxiv_url: str, save_folder: str) -> tuple:
    """
    Download a paper directly from an arXiv URL.

    Parses the URL, fetches metadata, downloads the PDF,
    and saves it with a clean filename.

    Args:
        arxiv_url:   Any arXiv URL or ID
        save_folder: Where to save the PDF

    Returns:
        Tuple of (pdf_path, metadata_dict)

    Raises:
        ValueError: If URL is not a valid arXiv link
    """
    # Extract arXiv ID
    arxiv_id = parse_arxiv_url(arxiv_url)
    print("arXiv ID: " + arxiv_id)

    # Fetch metadata (title, authors, etc.)
    metadata = fetch_paper_metadata(arxiv_id)
    print("Title: " + metadata["title"][:60])

    # Build clean filename from title
    clean_title = re.sub(r"[^a-zA-Z0-9 ]", "", metadata["title"])[:50]
    clean_title = clean_title.strip().replace(" ", "_")
    filename = arxiv_id + "_" + clean_title + ".pdf"

    # Download the PDF
    pdf_path = download_paper(
        pdf_url="https://arxiv.org/pdf/" + arxiv_id + ".pdf",
        save_folder=save_folder,
        filename=filename,
    )

    return pdf_path, metadata

# ─────────────────────────────────────────────────────────────
# Download a paper PDF to disk
# ─────────────────────────────────────────────────────────────

def download_paper(pdf_url: str, save_folder: str, filename: str = None) -> str:
    """
    Download a PDF from arXiv and save it to disk.

    Args:
        pdf_url:     Direct PDF URL (from search results)
        save_folder: Where to save the PDF
        filename:    Optional custom filename (default: arxiv ID)

    Returns:
        Full path to the saved PDF file

    Raises:
        Exception: If download fails
    """
    folder = Path(save_folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Generate filename from URL if not provided
    if not filename:
        arxiv_id = pdf_url.split("/")[-1].replace(".pdf", "")
        filename = f"{arxiv_id}.pdf"

    # Ensure .pdf extension
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    save_path = folder / filename

    print(f"Downloading: {filename}")

    # Stream download
    req = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": "ChatPaper-Research-Assistant/1.0"}
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        with open(save_path, "wb") as f:
            while True:
                chunk = response.read(8192)  # Read 8KB at a time
                if not chunk:
                    break
                f.write(chunk)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {save_path} ({file_size_mb:.1f} MB)")

    return str(save_path)