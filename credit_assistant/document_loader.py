from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    location: str
    text: str


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def read_docx_paragraphs(path: Path) -> list[str]:
    """Read text paragraphs from a DOCX without adding a heavy dependency."""
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")

    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", WORD_NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS))
        if text.strip():
            paragraphs.append(text.strip())
    return paragraphs


def read_pdf_pages(path: Path) -> list[str]:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "Pentru citirea PDF-urilor instaleaza PyPDF2 sau foloseste varianta DOCX."
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return [page for page in pages if page]


def paragraphs_from_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx_paragraphs(path)
    if suffix == ".pdf":
        return read_pdf_pages(path)
    raise ValueError(f"Format nesuportat: {path}")


def chunk_paragraphs(
    paragraphs: Iterable[str],
    source: str,
    *,
    max_chars: int = 1_600,
    overlap_paragraphs: int = 1,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    buffer: list[str] = []
    start_idx = 1

    for idx, paragraph in enumerate(paragraphs, start=1):
        candidate = "\n".join(buffer + [paragraph])
        if buffer and len(candidate) > max_chars:
            chunks.append(
                DocumentChunk(
                    source=source,
                    location=f"paragrafele {start_idx}-{idx - 1}",
                    text="\n".join(buffer),
                )
            )
            buffer = buffer[-overlap_paragraphs:] if overlap_paragraphs else []
            start_idx = max(1, idx - len(buffer))
        buffer.append(paragraph)

    if buffer:
        chunks.append(
            DocumentChunk(
                source=source,
                location=f"paragrafele {start_idx}-{start_idx + len(buffer) - 1}",
                text="\n".join(buffer),
            )
        )
    return chunks


def load_documents(paths: Iterable[Path]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    skipped: list[str] = []
    for path in paths:
        try:
            paragraphs = paragraphs_from_file(path)
        except RuntimeError as exc:
            skipped.append(f"{path.name}: {exc}")
            continue
        chunks.extend(chunk_paragraphs(paragraphs, path.name))
    if not chunks:
        raise ValueError("Nu am gasit continut text in documentele incarcate.")
    for message in skipped:
        chunks.append(
            DocumentChunk(
                source="Sistem",
                location="avertisment incarcare corpus",
                text=(
                    "Document omis la indexare. "
                    f"{message}. Ruleaza `python -m pip install -r requirements.txt` "
                    "si reporneste aplicatia pentru citirea PDF-urilor."
                ),
            )
        )
    return chunks
