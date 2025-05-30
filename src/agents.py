from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

def chat_completion(prompt, system_message="You are a helpful assistant.", model="gpt-4-1106-preview"):
    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def librarian_agent(state: dict) -> dict:
    kb = state.get("kb", "")
    question = state.get("question", "")
    
    prompt = (
        f"Based on the following Knowledge Base, select the most relevant chunks to answer the question. Be sure to select as few chunks as possible. Explicitly delimit the chunks you've found\n"
        f"\n[KNOWLEDGE BASE]\n{kb}\n"
        f"\n[QUESTION]\n{question}\n"
        f"\n[SELECTED CHUNKS]"
    )

    chunks = chat_completion(prompt, system_message='You are a helpful retriever bot.')
    return {**state, "chunks": chunks}


def editor_agent(state: dict) -> dict:
    chunks = state.get("chunks", "")
    question = state.get("question", "")
    
    prompt = (
        f"Based on the chunks below, you must give an answer in a clear and complete way to the following question. When the question is ambiguous and you've received different chunks containing the information about different visions, you MUST write a complete answer, giving the different possible nuances. For example: if the question is about growth and you've received chunks containing information of growth for different products, show all of them in your answer.\n"
        f"\n[CHUNKS]\n{chunks}\n"
        f"\n[QUESTION]\n{question}\n"
        f"\n[ANSWER]"
    )

    answer = chat_completion(prompt, system_message='You are a talented writer.')
    return {**state, "answer": answer}
