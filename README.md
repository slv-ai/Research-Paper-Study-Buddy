# Research-Paper-Study-Buddy
A Multi-turn Conversational Agent that helps you to study a arxiv research article

## Features
    1.Paper Summarization
        - Provide an arXiv URL or ID.
        - Get a summary of the paper and a list of prerequisites needed to understand it.

    2.Contextual Q&A
        - Ask questions about a paper.
        - The agent searches relevant chunks in the vector store and answers using only the paper content.

````
pip install uv
uv add openai requests pydantic-ai arxiv pydantic 
uv add chromadb sentence-transformers tiktoken
```
