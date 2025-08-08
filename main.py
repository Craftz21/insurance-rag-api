import os
import requests
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Header
from pydantic import BaseModel
from PyPDF2 import PdfReader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ------------------ ENV + CONFIG ------------------
load_dotenv()

def require_env(var_name: str) -> str:
    val = os.getenv(var_name)
    if not val:
        raise RuntimeError(f"❌ Missing required environment variable: {var_name}")
    return val

GROQ_API_KEY = require_env("GROQ_API_KEY")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-70b-8192"

PROMPT_TEMPLATE = """
Answer the question using only the information provided in the context.
Do not guess. If the answer is not in the context, respond with "I don't know."

Context:
{context}

Question: {query}
Answer:
"""

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
llm = ChatGroq(model_name=LLM_MODEL, api_key=GROQ_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ------------------ FASTAPI ------------------
app = FastAPI()

class Question(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Groq-powered insurance RAG API is running."}

@app.post("/ask")
def ask_question(q: Question):
    dummy_docs = [
        Document(page_content="Bajaj Allianz is headquartered in Pune."),
        Document(page_content="Global Health Care policy covers medical treatment outside India."),
        Document(page_content="Group Domestic Travel Insurance is provided by Cholamandalam MS General Insurance Company Limited."),
        Document(page_content="The UIN of the Bajaj Allianz policy is UIN- BAJHLIP23020V012223.")
    ]
    docs = splitter.split_documents(dummy_docs)
    vectordb = FAISS.from_documents(docs, embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    rel_docs = retriever.invoke(q.query)
    context = "\n\n---\n\n".join([doc.page_content for doc in rel_docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context,
        query=q.query,
    )
    response = llm.invoke(prompt)

    sources = [
        {
            "source": doc.metadata.get("source", "dummy"),
            "content": doc.page_content[:200],
        }
        for doc in rel_docs
    ]

    return {
        "answer": response.content if hasattr(response, "content") else str(response),
        "sources": sources,
    }

@app.post("/debug-retrieval")
def debug_retrieval(q: Question):
    dummy_docs = [
        Document(page_content="Bajaj Allianz is headquartered in Pune."),
        Document(page_content="Global Health Care policy covers medical treatment outside India."),
        Document(page_content="Group Domestic Travel Insurance is provided by Cholamandalam MS General Insurance Company Limited."),
        Document(page_content="The UIN of the Bajaj Allianz policy is UIN- BAJHLIP23020V012223.")
    ]
    docs = splitter.split_documents(dummy_docs)
    vectordb = FAISS.from_documents(docs, embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    rel_docs = retriever.invoke(q.query)
    return {
        "retrieved": [
            {
                "source": doc.metadata.get("source", "dummy"),
                "content": doc.page_content[:500],
            }
            for doc in rel_docs
        ]
    }

# ------------------ HACKRX ENDPOINT ------------------

class HackRxInput(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def hackrx_run(body: HackRxInput, authorization: str = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        return {"success": False, "error": "Missing or invalid Authorization header"}

    pdf_url = body.documents
    questions = body.questions

    try:
        pdf_response = requests.get(pdf_url)
        pdf_response.raise_for_status()
        pdf_reader = PdfReader(BytesIO(pdf_response.content))
        full_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
    except Exception as e:
        return {"success": False, "error": f"PDF download/parse failed: {str(e)}"}

    try:
        documents = [Document(page_content=full_text)]
        split_docs = splitter.split_documents(documents)
        vectordb = FAISS.from_documents(split_docs, embedding)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        return {"success": False, "error": f"Vector index creation failed: {str(e)}"}

    answers = []
    for query in questions:
        try:
            rel_docs = retriever.invoke(query)
            context = "\n\n---\n\n".join([doc.page_content for doc in rel_docs])
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
                context=context,
                query=query,
            )
            result = llm.invoke(prompt)
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            answer = f"Error answering: {str(e)}"
        answers.append(answer)

    return {"success": True, "answers": answers}
