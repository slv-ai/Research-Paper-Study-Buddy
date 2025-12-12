from dataclasses import dataclass
from tools.paper_processor import PaperProcessor
from tools.vector_chunks import VectorChunks
from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic import BaseModel, Field

vector_store = VectorChunks()
paper_processor = PaperProcessor()

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

def process_and_summarize(ctx: RunContext, file_path: str) -> str:
    """
    A single tool that:
    1. Loads and processes a document and extracts text and metadata
    2. Splits into chunks for embedding
    3. Stores chunks in vector DB
    4. Summarizes the content based on summarize instructions

    Returns: Final summary text.
    """
    # Step 1: Load and process document
    paper_metadata = paper_processor.fetch_paper(file_path)
    pdf_path = paper_processor.download_pdf(paper_metadata.pdf_url)
    pages = paper_processor.extract_text_from_pdf(pdf_path)

    # Step 2: Split into chunks
    chunks = paper_processor.chunk_paper(pages, paper_metadata)

    # Step 3: Store chunks in vector DB
    vector_store.add_paper_chunks(chunks,metadata)

    # Step 4: Summarize content
    summarize_instructions = """
    You are a friendly and knowledgeable study buddy helping someone understand a research paper
    Your approach:
- Be conversational and encouraging
- Explain concepts clearly with examples
- Reference specific parts of the paper when answering
- Ask clarifying questions when needed
- Suggest related concepts to explore
- Break down complex ideas into simple terms
- Use analogies to make things relatable

When answering:
- Base your answers on the paper content provided
- If the paper doesn't contain the answer, say so
- Cite which section/page you're referencing
- Offer to explain prerequisites if needed"""
    

    return summary

agent = Agent(
        name="summarizer_agent",
        instructions=summarize_instructions,
        tools=[process_and_summarize],
        model="gpt-4o-mini",
        
    )