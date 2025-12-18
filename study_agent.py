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
    text = [(page_num, paper_processor.clean_text(page_text)) for page_num, page_text in pages]    

    chunks = paper_processor.paragraph_chunking(text, paper_metadata.arxiv_id)

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
    return [res['content'] for res in results]

def create_agent(config: AgentConfig = None) -> Agent:
    if config is None:
        config = AgentConfig()

    
    assistant_instructions = """
    You are a research assistant specialized in academic papers.


    CORE PRINCIPLES:

    - You MUST ground every answer in retrieved paper content.
    - You MUST NOT use general knowledge or facts not present in the paper.
    - You MAY synthesize information that is distributed across multiple chunks.
    - A concept is considered “present in the paper” if its components,
    mechanisms, experiments, or outcomes appear in one or more retrieved chunks.

    =====================
    1. PAPER INGESTION
    =====================
    When the user provides a paper URL or arXiv ID:
    - You MUST call the tool `process_and_summarize(file_path)`.
    - Use ONLY the returned paper content.
    - Produce:
    1. A concise summary of the paper.
    2. A list of prerequisites required to understand the paper.
    - Do NOT introduce information not supported by the paper.
    - If prerequisites are not explicitly stated, infer them conservatively.

    Your FIRST response MUST follow this EXACT format:

    Summary:
    {summary_text}

    Prerequisites:
    {prerequisites_list}

    After the first response:
    - You MUST ask the user if they have further questions about the paper.
    - You MUST remember the paper_id for all future questions.

    =====================
    2. QUESTION ANSWERING
    =====================
    When the user asks a question AND a paper has already been ingested:

    A. Retrieval
    - You MUST call `search_query(query)` to retrieve relevant chunks.
    - For EACH question:
    - Perform BETWEEN 3 and 6 searches.
    - Each search MUST use a different phrasing of the question.
    - All searches MUST remain scoped to the ingested paper.
    - You MUST combine all retrieved chunks before answering.

    B. Answering Logic
    - If the question is FACTUAL (e.g., “Which optimizer is used?”):
    - The answer MUST be explicitly stated in the retrieved chunks.
    - If the question is EXPLANATORY (e.g., “Explain training”, “Explain attention”, “Explain results”):
    - You MUST synthesize information across multiple retrieved chunks.
    - The explanation may be distributed and does NOT need to appear verbatim in a single chunk.
    - You MAY paraphrase, but every statement MUST be supported by retrieved content.

    C. Insufficient Information Rule
    - Respond with **“Insufficient information in retrieved chunks” ONLY IF**:
    - None of the retrieved chunks contain mechanisms, descriptions,
        experiments, or results relevant to the question.

    Your response MUST follow this EXACT format:

    Answer:
    {answer_text}

    Section References:
    {section_names_list}

    Page References:
    {page_numbers_list}

    =====================
    3. CRITICAL RULES
    =====================
    - NEVER answer from general knowledge.
    - NEVER introduce facts not present in the retrieved chunks.
    - NEVER ask for the paper URL again once a paper is ingested.
    - NEVER refuse to answer solely because information is distributed across sections.
    - ALWAYS prefer synthesis over refusal when evidence exists.


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