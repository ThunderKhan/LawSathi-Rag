import logging
import re
from typing import List, Optional, Tuple

import tiktoken
from tqdm import tqdm

logger = logging.getLogger(__name__)

_LEGAL_HEADING_PREFIX = re.compile(
    r"^\s*(?:sections?|sec\.?|articles?|art\.?|chapters?|chap\.?|parts?|"
    r"schedules?|rules?|regulations?|orders?|clauses?|sub[- ]?sections?)\b"
    r"[^\n]{0,120}$",
    re.IGNORECASE,
)
_ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9\s,./&()'’:-]{1,79}$")


def _is_legal_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False

    if _LEGAL_HEADING_PREFIX.match(stripped):
        return True

    if _ALL_CAPS_HEADING.fullmatch(stripped):
        return any(char.isalpha() for char in stripped) and len(stripped.split()) <= 10

    return False


def _split_legal_sections(text: str) -> List[Tuple[Optional[str], str]]:
    sections: List[Tuple[Optional[str], str]] = []
    current_heading: Optional[str] = None
    current_lines: List[str] = []

    for line in text.splitlines():
        if _is_legal_heading(line):
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((current_heading, section_text))
            current_heading = line.strip()
            current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_heading, section_text))
    elif current_heading:
        sections.append((current_heading, ""))

    return sections


def _chunk_token_window(text: str, enc, chunk_size: int, overlap: int) -> List[str]:
    tokens = enc.encode(text)
    if len(tokens) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_str = enc.decode(tokens[start:end]).strip()
        if chunk_str:
            chunks.append(chunk_str)
        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def _chunk_legal_section(
    heading: Optional[str],
    body: str,
    enc,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    body = body.strip()

    if not heading:
        return _chunk_token_window(body, enc, chunk_size, overlap)

    if not body:
        return [heading]

    heading_tokens = enc.encode(heading)
    if len(heading_tokens) >= chunk_size:
        return _chunk_token_window(f"{heading}\n{body}".strip(), enc, chunk_size, overlap)

    body_tokens = enc.encode(body)
    body_chunk_size = chunk_size - len(heading_tokens)
    if len(body_tokens) <= body_chunk_size:
        return [f"{heading}\n{body}".strip()]

    chunks: List[str] = []
    start = 0
    step = body_chunk_size - overlap

    if step <= 0:
        step = body_chunk_size

    while start < len(body_tokens):
        end = min(start + body_chunk_size, len(body_tokens))
        body_chunk = enc.decode(body_tokens[start:end]).strip()
        if body_chunk:
            chunks.append(f"{heading}\n{body_chunk}".strip())
        if end >= len(body_tokens):
            break
        start += step

    return chunks


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split legal judgment text into token-based chunks while preserving explicit
    legal section and heading boundaries where possible.

    Args:
        text (str): Input text to split.
        chunk_size (int): Max token size for each chunk.
        overlap (int): Number of overlapping tokens between adjacent chunks.

    Returns:
        List[str]: Cleaned list of text chunks.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        enc = tiktoken.get_encoding("cl100k_base")
        stripped_text = text.strip()

        sections = _split_legal_sections(stripped_text)
        if not sections:
            return _chunk_token_window(stripped_text, enc, chunk_size, overlap)

        chunks: List[str] = []
        for heading, body in sections:
            chunks.extend(_chunk_legal_section(heading, body, enc, chunk_size, overlap))

        return [chunk for chunk in chunks if chunk.strip()]
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return [text.strip()] if text.strip() else []


def chunk_documents(documents: List[str], chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split a list of documents into a flattened list of token-based chunks with progress tracking.

    Args:
        documents (List[str]): List of document strings.
        chunk_size (int): Target chunk size in tokens.
        overlap (int): Token overlap.

    Returns:
        List[str]: Flattened list of non-empty chunks.
    """
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking documents"):
        try:
            chunks = chunk_text(doc, chunk_size, overlap)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing document: {e}")
    return all_chunks
