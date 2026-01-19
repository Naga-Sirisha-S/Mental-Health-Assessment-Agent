import os
import random
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document


# Load environment and setup

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

df = pd.read_excel("ques_ans.xlsx")

# Convert rows to documents
docs = []
for _, row in df.iterrows():
    text = f"Question: {row['Question']}, Answer: {row['Answer']}"
    docs.append(Document(page_content=text))

# Chunking
text_splitter = CharacterTextSplitter(chunk_size=80, chunk_overlap=20)
docs_split = text_splitter.split_documents(docs)

# Embeddings + Vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    docs_split, embedding=embeddings, persist_directory="chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# LLM Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="AI Doctor Chatbot")

class MessageInput(BaseModel):
    message: str

empathy_instruction = """
You are a compassionate AI Doctor specializing in trauma-aware mental health and sleep disorders.  
Guidelines:
- Respond with warmth and empathy.
- Provide supportive psychoeducation when relevant.
- Occasionally include ONE follow-up question from the provided list.
- If patient seems very distressed, prioritize calm reassurance and grounding.
"""

doctor_questions = [
    "Can you tell me in your own words how you’ve been feeling lately?",
    "Do you often feel nervous, restless, or on edge?",
    "Have there been any recent situations or stressors that triggered these feelings?",
    "Are there certain things or situations that make you feel especially anxious or uncomfortable?",
    "Have you experienced any sudden moments of intense fear or panic, even when there wasn’t a clear reason?",
    "Do you ever feel like something terrible is about to happen, even if nothing is obviously wrong?",
    "Do your thoughts sometimes race or jump quickly from one worry to another?",
    "Do you find it hard to stop worrying, even when you try to relax?",
    "Have you had thoughts that feel frightening, irrational, or hard to control?",
    "Do you ever feel detached from yourself or your surroundings like things don’t feel real?"
]

distress_keywords = ["panic", "can’t sleep", "cannot sleep", "very anxious", "heart racing", "overwhelmed"]

conversation_log = []

@app.post("/chat")
async def chat(input: MessageInput):
    user_message = input.message
    conversation_log.append({"role": "patient", "content": user_message})

    # Retrieve context
    related_docs = retriever.get_relevant_documents(user_message)
    retrieved_context = " ".join([doc.page_content for doc in related_docs])

    # Distress detection
    distress_flag = any(word in user_message.lower() for word in distress_keywords)

    # Follow-up question (50% chance if not distressed)
    follow_up_question = ""
    if not distress_flag and random.random() < 0.5:
        follow_up_question = random.choice(doctor_questions)

    doctor_prompt = f"""
The patient said: {user_message}
Here is related knowledge: {retrieved_context}

Respond empathetically in 2–4 sentences:
- Validate their feelings warmly.
- Offer reassurance or simple advice.
- If included, append this follow-up clinical question naturally at the end: "{follow_up_question}"
"""

    response = llm.invoke([
        {"role": "system", "content": empathy_instruction},
        {"role": "user", "content": doctor_prompt}
    ])

    reply_text = response.content
    conversation_log.append({"role": "doctor", "content": reply_text})

    return {"reply": reply_text}


@app.get("/final_reflection")
async def final_reflection():
    all_answers = " ".join([f"Patient: {turn['content']}" for turn in conversation_log if turn['role'] == "patient"])
    all_context = " ".join([turn['content'] for turn in conversation_log if turn['role'] == "doctor"])

    final_prompt = f"""
Here is the patient's overall conversation:
{all_answers}

Here are the doctor's responses:
{all_context}

Now provide a final compassionate summary in 3–5 sentences.
- Validate their feelings.
- Normalize their experiences.
- Suggest a calming strategy (breathing, grounding).
- Encourage ongoing self-care or professional support.
"""

    final_response = llm.invoke([
        {"role": "system", "content": empathy_instruction},
        {"role": "user", "content": final_prompt}
    ])

    final_message = final_response.content
    conversation_log.append({"role": "doctor", "content": final_message})

    return {"reflection": final_message}



# Run the server

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
