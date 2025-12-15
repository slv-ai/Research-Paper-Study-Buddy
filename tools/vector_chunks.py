from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

class PaperChunk(BaseModel):
    """A chunk of paper content"""
    chunk_id: str
    paper_id: str
    content: str
    section: str  # "abstract", "introduction", "methods", etc.
    chunk_index: int
    page_number: int

class PaperMetadata(BaseModel):
    """Paper metadata"""
    arxiv_id: str
    title: str
    authors: List[str]
    published_date: str
    abstract: str
    pdf_url: str

class VectorStore:
    """Manages paper chunks in a vector database"""
    def __init__(self,persist_directory: str = "./chroma_db"):

        """initalize chromaDB"""
        
        self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
        ))

        #create or get collection
        self.collection = self.client.get_or_create_collection(
            name="arxiv_papers",
            metadata={"description": "arxiv paper chunks"}
        )

        #embeddinng model
           
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Vector store initialized with {self.collection.count()} chunks")
        
    def add_paper_chunks(self, chunks: List[PaperChunk], metadata: PaperMetadata):
        """add paper chunks to vector store"""
        if not chunks:
            return
        
        # Filter out invalid chunks
        valid_chunks = [chunk for chunk in chunks if chunk.content and chunk.content.strip()]
        if not valid_chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in valid_chunks]
        embeddings = self.embedding_model.encode(texts).tolist()  # safe now
        
        # Prepare metadata
        metadatas = [
            {
                "paper_id": chunk.paper_id,
                "paper_title": metadata.title,
                "section": chunk.section,
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index
            }
            for chunk in valid_chunks
        ]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=[chunk.chunk_id for chunk in valid_chunks]
        )
        print(f"Added {len(valid_chunks)} chunks to vector store")
       


    def search_relevant_chunks(self, query: str,  n_results: int = 15):
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            #where={"paper_id": paper_id}
        )
        # ===== TEMP DEBUG LOGGING =====
        print("\n[DEBUG] Raw retrieval results")
        print("Query:", query)
        print("Number of hits:", len(results["ids"][0]))

        for i in range(len(results["ids"][0])):
            print(f"\n--- HIT {i} ---")
            print("Chunk ID:", results["ids"][0][i])
            print("Section:", results["metadatas"][0][i].get("section"))
            print("Page:", results["metadatas"][0][i].get("page"))
            print("Distance:", results["distances"][0][i])
            print("Content preview:")
            print(results["documents"][0][i][:400])
        # ===== END DEBUG =====

        relevant_chunks = []
        query_lower = query.lower()

        for i in range(len(results["ids"][0])):
            content = results["documents"][0][i]

            # keyword boost
            keyword_score = sum(
                1 for word in query_lower.split()
                if word in content.lower()
            )

            relevant_chunks.append({
                "chunk_id": results["ids"][0][i],
                "content": content,
                "section": results["metadatas"][0][i]["section"],
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i],
                "keyword_score": keyword_score
            })

        # Re-rank by keyword_score first, then distance
        relevant_chunks.sort(
            key=lambda x: (-x["keyword_score"], x["distance"])
        )

        return relevant_chunks[:n_results]


    def delete_paper(self, paper_id: str):
        """Remove all chunks for a paper"""
        # Query all chunks for this paper
        results = self.collection.get(where={"paper_id": paper_id})
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks for paper {paper_id}")


