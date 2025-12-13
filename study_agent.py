from dataclasses import dataclass
from tools.paper_processor import PaperProcessor
from tools.vector_chunks import VectorStore
from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic import BaseModel, Field
from typing import List

vector_store = VectorStore()
paper_processor = PaperProcessor()


class AgentConfig:
    model: str = "openai:gpt-4o-mini"

class NamedCallback:

    def __init__(self, agent):
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        # Detect nested streams
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)

def process_and_summarize(file_path: str) -> str:
    """Process paper, store chunks, return summary + prerequisites."""
  
    # Step 1: Fetch metadata and download PDF
    paper_metadata = paper_processor.fetch_paper(file_path)
    pdf_path = paper_processor.download_pdf(paper_metadata.pdf_url)

    # Step 2:  Extract text + chunking
    pages = paper_processor.extract_text_from_pdf(pdf_path)    
    chunks = paper_processor.chunk_paper(pages, paper_metadata.arxiv_id)

    # Step 3: Store chunks in vector DB
    vector_store.add_paper_chunks(chunks,paper_metadata)

    # Step 4: Summarize content
    full_text = "\n\n".join([c.content for c in chunks]) 
       
    return full_text

def search_query(query: str) -> List[str]:
    """Search for relevant paper chunks given a query"""
    results = vector_store.search_relevant_chunks(query,PaperChunk.paper_id, n_results=5)
    return [res['document'] for res in results]

def create_agent(config: AgentConfig = None) -> Agent:
    if config is None:
        config = AgentConfig()

    
    assistant_instructions = """
    You are a research assistant specialized in academic papers.

    Behavior rules:

    1. **Paper ingestion**
    - When the user provides a paper URL or arXiv ID, you must:
        - Use the `process_and_summarize(file_path)` tool.
        - Return a concise **summary** of the paper.
        - List **prerequisites** the reader should know to understand the paper.
        - Do not use external knowledge—base the summary on the paper content.
        - summary guidelines: Guidelines:
        - Write clearly and concisely.
        - Focus on the paper’s main contributions, methods, and findings.
        - Do NOT introduce information that is not present in the paper.
        - If prerequisites are not explicitly stated, infer them conservatively.
        - Use simple, student-friendly language.
        Your first response MUST follow this exact format:

        Summary:
        {summary_text}

        Prerequisites:
        {prerequisites_list}
    2. **Question answering**
    - When the user asks a question about a paper:
        - Use the `search_chunks(query, paper_id)` tool to retrieve relevant chunks.
        - Answer the question **only using information from these chunks**.
        - Include section and page numbers when citing information.
        - If the chunks do not contain enough information, reply: "Insufficient information in retrieved chunks."
        - Do not rely on general knowledge or memorized facts.

    Tools available:
    - `process_and_summarize(file_path)`: Ingests a paper and stores its chunks.
    - `search_chunks(query)`: Retrieves the most relevant chunks for a query.
    """
    

                
    agent = Agent(
        name="study_agent",
        instructions=assistant_instructions,
        tools=[process_and_summarize,search_query],
        model=config.model,
        
    )
    return agent