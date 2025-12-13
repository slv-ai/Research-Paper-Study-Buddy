import os
import re
import json
import uuid
import PyPDF2
import arxiv
import io
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field
import tiktoken


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
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def fetch_paper(self, arxiv_id: str) -> PaperMetadata:
        """Fetch paper metadata"""
        arxiv_id = self.extract_arxiv_id(arxiv_id)
        
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=paper.title,
            authors=[author.name for author in paper.authors],
            published_date=paper.published.strftime('%Y-%m-%d'),
            abstract=paper.summary,
            pdf_url=paper.pdf_url
        )

    def download_pdf(self, pdf_url: str, save_path: str = "temp_paper.pdf") -> str:
        """Download PDF"""
        import requests
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path

    def extract_text_from_pdf(self,pdf_path: str) -> List[tuple]:
        """ download pdf and extract text """
        
        #extract text
        reader = PyPDF2.PdfReader(pdf_path)
        pages = []
            
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append((i + 1, text))
            
        return pages
    
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
                return match.group(1).replace('v', '.')
        
        return input_str.strip()
    
    def chunk_paper(
        self, 
        pages: List[tuple], 
        paper_id: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[PaperChunk]:
        """Create semantic chunks from paper"""
        chunks = []
        chunk_index = 0
        for page_num, page_text in pages:
            # Detect section
            section = self.detect_section(page_text)
            
            # Split page into chunks with overlap
            tokens = self.tokenizer.encode(page_text)
            
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                # Skip very short chunks
                if len(chunk_text.strip()) < 100:
                    continue
                
                chunks.append(PaperChunk(
                    chunk_id=f"{paper_id}_chunk_{chunk_index}",
                    paper_id=paper_id,
                    content=chunk_text,
                    section=section,
                    chunk_index=chunk_index,
                    page_number=page_num
                ))
                
                chunk_index += 1
        
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

