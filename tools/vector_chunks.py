from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
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
        
        #generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts).tolist()
        # Prepare metadata
        metadatas = [
            {
                "paper_id": chunk.paper_id,
                "paper_title": metadata.title,
                "section": chunk.section,
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index
            }
            for chunk in chunks
        ]
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=[chunk.chunk_id for chunk in chunks]
        )
        print(f"Added {len(chunks)} chunks to vector store")

        def search_relevant_chunks(self,query: str,paper_id: str,n_results: int = 5) -> List[Dict[str, Any]]:
            """Search for relevant chunks given a query"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"paper_id": paper_id}
        )

        # Format results
        relevant_chunks = []
        for i in range(len(results['ids'][0])):
            relevant_chunks.append({
                'chunk_id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'section': results['metadatas'][0][i]['section'],
                'page': results['metadatas'][0][i]['page'],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return relevant_chunks

 def delete_paper(self, paper_id: str):
        """Remove all chunks for a paper"""
        # Query all chunks for this paper
        results = self.collection.get(where={"paper_id": paper_id})
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks for paper {paper_id}")


