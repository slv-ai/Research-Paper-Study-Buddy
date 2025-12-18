import study_agent
import asyncio
from tools.paper_processor import PaperProcessor
from pydantic import BaseModel

agent = study_agent.create_agent()
agent_callback = study_agent.NamedCallback(agent)
current_id_processor= PaperProcessor()


async def run_agent(user_prompt: str):
    results = await agent.run(
        user_prompt=user_prompt,
        event_stream_handler=agent_callback
    )

    return results


def run_agent_sync(user_prompt: str):
    return asyncio.run(run_agent(user_prompt))

current_paper_id = None
while True:
    user_input = input("Enter arXiv URL or ask a question (exit to quit): ").strip()
    if user_input.lower() == "exit":
        break

    # CASE 1: User provides paper URL
    if current_id_processor.is_arxiv_url(user_input):
        result = run_agent_sync(user_input)
        current_paper_id = current_id_processor.extract_arxiv_id(user_input)
        print(result.output)
        #print("\nYou can now ask questions about this paper.\n")
        continue

    
    result = run_agent_sync(user_input)
    print(result.output)
