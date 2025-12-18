import os
import re
import json
import uuid
import fitz
import arxiv
import io
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field



class PaperMetadata(BaseModel):
    """Paper metadata"""
    arxiv_id: str
    title: str
    authors: List[str]
    published_date: str
    abstract: str
    pdf_url: str

class PaperChunk(BaseModel):
    """A chunk of paper content"""
    chunk_id: str
    paper_id: str
    content: str
    section: str  # "abstract", "introduction", "methods", etc.
    chunk_index: int
    page_number: int

class PaperProcessor:
    """ process papers and create chunks for embedding """
    def __init__(self):
        pass        
    
    def fetch_paper(self, arxiv_id: str) -> PaperMetadata:
        arxiv_id = self.extract_arxiv_id(arxiv_id)
        search = arxiv.Search(id_list=[arxiv_id])

        try:
            paper = next(search.results())
        except StopIteration:
            raise ValueError(f"No paper found for arXiv ID {arxiv_id}")

        published = (
            paper.published.strftime('%Y-%m-%d')
            if paper.published else ""
        )

        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=paper.title,
            authors=[a.name for a in paper.authors],
            published_date=published,
            abstract=paper.summary,
            pdf_url=paper.pdf_url
        )


    def download_pdf(self, pdf_url: str, save_path="temp_paper.pdf") -> str:
        if not pdf_url:
            raise ValueError("PDF URL is empty")

        import requests
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return save_path


    def extract_text_from_pdf(self, pdf_path: str) -> List[tuple]:
        doc = fitz.open(pdf_path)
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text().strip()

            if not text:
                blocks = page.get_text("blocks")
                text = "\n".join(
                    b[4] for b in blocks
                    if len(b) > 4 and isinstance(b[4], str)
                )

            pages.append((i + 1, text))

        doc.close()
        return pages

        
        
    
    def is_arxiv_url(self, text : str) -> bool:
        """Check if text is an arXiv URL or ID"""
        text = text.strip()

        # arXiv URL
        if "arxiv.org/abs/" in text or "arxiv.org/pdf/" in text:
            return True

        # arXiv ID like 1706.03762 or 1706.03762v1
        arxiv_id_pattern = r"^\d{4}\.\d{4,5}(v\d+)?$"
        if re.search(arxiv_id_pattern, text):
            return True

        return False
    
    def extract_arxiv_id(self, input_str: str) -> str:
        """Extract ArXiv ID"""
        patterns = [
            r'arxiv\.org/abs/(\d+\.\d+)',
            r'arxiv\.org/pdf/(\d+\.\d+)',
            r'^(\d+\.\d+v?\d*)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, input_str)
            if match:
                return match.group(1)
        
        return input_str.strip()
    
    def simple_overlap_chunking(self, pages: List[tuple], paper_id: str, chunk_size: int = 800, overlap: int = 200) -> List[PaperChunk]:
        chunks = []
        chunk_index = 0
        for page_num, page_text in pages:
            if not page_text or len(page_text.strip()) < 50:
                continue
            start = 0
            text_length = len(page_text)
            while start < text_length:
                end = start + chunk_size
                chunk_text = page_text[start:end]
                if len(chunk_text.strip()) >= 150 :
                    chunks.append(PaperChunk(
                        chunk_id=f"{paper_id}_chunk_{chunk_index}",
                        paper_id=paper_id,
                        content=chunk_text,
                        section = "content",
                        chunk_index=chunk_index,
                        page_number=page_num
                    ))
                    chunk_index += 1
                start += chunk_size - overlap

        return chunks

    import re

    def clean_text(self, text: str) -> str:
        """Clean extracted text for RAG ingestion."""

        # Remove token artifacts (<pad>, <EOS>, etc.)
        text = re.sub(r"<\s*(pad|EOS|eos)\s*>", " ", text)

        # Collapse weird spacing
        text = re.sub(r"\s+", " ", text)

        # Remove figure & table captions
        text = re.sub(r"Figure\s*\d+[:\.]\s*.*?(?=Figure\s*\d+[:\.]|$)", " ", text)
        text = re.sub(r"Table\s*\d+[:\.]\s*.*?(?=Table\s*\d+[:\.]|$)", " ", text)

        # Remove page headers/footers that repeat every page (heuristic)
        text = re.sub(r"^Page\s*\d+\s*", " ", text, flags=re.MULTILINE)

        # Remove lines with too many single letters (e.g., token dumps)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            tokens = line.strip().split()
            short_tokens = sum(1 for t in tokens if len(t) == 1)
            if len(tokens) > 0 and short_tokens / len(tokens) > 0.4:
                continue
            cleaned_lines.append(line)
        text = " ".join(cleaned_lines)

        # Collapse whitespace again
        text = re.sub(r"\s+", " ", text).strip()

        return text


    def paragraph_chunking(self, pages: List[tuple], paper_id: str) -> List[PaperChunk]:
        "paragraph based chunking"
        chunks=[]
        current_chunk=""
        chunk_idx=0
        for page_num,page_text in pages:
            paragraphs=[p.strip() for p in page_text.split('\n\n') if len(p.strip()) > 50]
            for para in paragraphs:
                if len(current_chunk) + len(para) > 1000:
                    chunks.append(PaperChunk(
                        chunk_id=f"{paper_id}_chunk_{chunk_idx}",
                        paper_id=paper_id,
                        content=current_chunk.strip(),
                        section = "content",
                        chunk_index=chunk_idx,
                        page_number=page_num
                    ))
                    chunk_idx += 1
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para
        if current_chunk:
                chunks.append(PaperChunk(
                    chunk_id=f"{paper_id}_chunk_{chunk_idx}",
                    paper_id=paper_id,
                    content=current_chunk.strip(),
                    section = "content",
                    chunk_index=chunk_idx,
                    page_number=page_num
                ))
        return chunks


    def detect_section(self, text: str) -> str:
        """Detect paper section from text"""
        text_lower = text.lower()
        
        sections = {
            'abstract': ['abstract'],
            'introduction': ['introduction', '1. introduction', '1 introduction'],
            'related_work': ['related work', 'background', 'literature review'],
            'methodology': ['methodology', 'methods', 'approach', 'model'],
            'experiments': ['experiments', 'experimental', 'evaluation'],
            'results': ['results', 'findings'],
            'discussion': ['discussion', 'analysis'],
            'conclusion': ['conclusion', 'concluding'],
            'references': ['references', 'bibliography']
        }
        
        for section, keywords in sections.items():
            if any(kw in text_lower[:500] for kw in keywords):
                return section
        
        return 'content'

    


# if __name__ == "__main__":
#     processor=PaperProcessor()
#     metadata = processor.fetch_paper("1506.07917")
#     pdf_path = processor.download_pdf(metadata.pdf_url)
#     pages = processor.extract_text_from_pdf(pdf_path)
#     for page_num, text in pages:
#         print(f"Page {page_num}:\n{text[:500]}...\n")
#     chunks = processor.chunk_paper(pages, metadata.arxiv_id)
#     for chunk in chunks[:3]:
#         print(f"Chunk {chunk.chunk_index} (Page {chunk.page_number}, Section: {chunk.section}):\n{chunk.content[:500]}...\n")

