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

def search_query(prompt: str) -> list[str]:
    """Search all chunks in the vector store without filtering by paper"""
    query = prompt.strip()
    
    if not query:
        raise ValueError("Query is empty. Please provide a valid question.")

    results = vector_store.search_relevant_chunks(
        query=query,
        n_results=15  # no paper_id filter
    )
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
        MUST Remember the paper_id for future questions.
        YOU  MUST ASK THE USER for further questions about the paper ingested paper{title}  AFTER FIRST RESPONSE.
        FOR ANSWERING QUESTIONS, FOLLOW THE INSTRUCTIONS BELOW.

    2. **Question answering**
    -When the user asks a question AND a paper has already been ingested:
        When answering questions:
        -  Call `search_query(query, paper_id)` to retrieve relevant chunks.
        - FOR EVERY QUERY : PERFORM ATLEAST 3 AND ATMOST 6 SEARCHES TO RETRIEVE RELEVANT CHUNKS.
        - Each search MUST use different phrasings of the query to maximize coverage.
        -KEEP all searches RELEVANT ONLY TO THE PAPER WITH paper_id.
        - If the concept is described across multiple retrieved chunks,
        synthesize them into a single explanation.
        - You may paraphrase, but must stay faithful to the retrieved content.
        - Cite section names and page numbers where possible.
        - Only respond with "Insufficient information in retrieved chunks"
        if the concept is not discussed anywhere in the retrieved content.
        - Do not rely on general knowledge or memorized facts.
        Your response MUST follow this exact format:
        Answer:
        {answer_text}
        Section References:
        {section_references_list}
        Page References:
        {page_references_list}
    3. **General rules**
        CRITICAL RULES
        =====================
        - Never answer from general knowledge.
        - Never request a paper URL again if a paper is already loaded.
        - Do not explain concepts unless they appear in retrieved chunks.

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