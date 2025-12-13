import study_agent
import asyncio

agent = study_agent.create_agent()
agent_callback = study_agent.NamedCallback(agent)


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
    user_input = input("Enter arXiv ID, URL, or question: ")

    if user_input.lower() == "exit":
        break

    if is_arxiv_or_url(user_input):
        # INGEST
        result = run_agent_sync(user_input)
        current_paper_id = extract_arxiv_id(user_input)
        print(result.output)

    else:
        # QUESTION
        if current_paper_id is None:
            print("Please ingest a paper first.")
            continue

        question_prompt = f"QUESTION::{current_paper_id}::{user_input}"
        result = run_agent_sync(question_prompt)
        print(result.output)

import re

def is_arxiv_or_url(text: str) -> bool:
    text = text.strip()

    # arXiv URL
    if "arxiv.org/abs/" in text or "arxiv.org/pdf/" in text:
        return True

    # arXiv ID like 1706.03762 or 1706.03762v1
    arxiv_id_pattern = r"^\d{4}\.\d{4,5}(v\d+)?$"
    if re.match(arxiv_id_pattern, text):
        return True

    return False

   


