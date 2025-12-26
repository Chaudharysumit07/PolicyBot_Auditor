import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser

# Internal import
from misc import utils

# --- Configuration ---
# Use a global variable to cache the embeddings model
EMBEDDINGS_MODEL = None


def get_embeddings_model():
    """
    Initializes and caches the embeddings model.
    """
    global EMBEDDINGS_MODEL
    if EMBEDDINGS_MODEL is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device} for embeddings.")

        EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs={"device": device}
        )
    return EMBEDDINGS_MODEL


async def create_vector_store(
    file_paths: List[Path],
    db_path: Path,
    client_id: str,
    websocket_manager: utils.ConnectionManager,
):
    """
    Loads, splits, and embeds documents, then saves them to a FAISS vector store.
    Sends progress updates via the WebSocket manager.
    """

    # 1. Load the documents
    # We use the parent directory of the files
    await websocket_manager.send_status_update(
        client_id, "parsing", f"Loading {len(file_paths)} documents..."
    )
    loader = DirectoryLoader(
        file_paths[0].parent,
        glob="**/*.*",
        show_progress=False,  # We send our own progress
        use_multithreading=True,
        loader_cls=UnstructuredFileLoader,
    )
    docs = loader.load()
    if not docs:
        raise ValueError("No documents were loaded. Check file types and paths.")

    await websocket_manager.send_status_update(
        client_id, "parsing", f"✅ {len(docs)} documents loaded."
    )

    # 2. Split the documents
    await websocket_manager.send_status_update(
        client_id, "chunking", "Splitting documents into chunks..."
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    await websocket_manager.send_status_update(
        client_id, "chunking", f"✅ Documents split into {len(splits)} chunks."
    )

    # 3. Create embeddings and vector store
    await websocket_manager.send_status_update(
        client_id, "embedding", "Creating embeddings (this may take a while)..."
    )
    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # 4. Save the vector store to disk
    vectorstore.save_local(str(db_path))
    await websocket_manager.send_status_update(
        client_id, "embedding", "✅ Vector store created and saved."
    )

    # 5. Create and return the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

'''
"""
    **System:** You are an expert cybersecurity GRC auditor. Your task is to answer the assessment question based ONLY on the provided context from the uploaded document.
    - Your response MUST be in the following JSON format: {{"answer": "Yes/No/Partial", "reasoning": "Your brief explanation.", "evidence": "The exact quote from the context."}}
    - Answer 'Yes' only if the context fully and explicitly satisfies the question.
    - Answer 'No' if the context does not mention the topic.
    - Answer 'Partial' if the topic is mentioned but lacks sufficient detail or doesn't meet the full requirement of the question.
    - If the context is insufficient, your reasoning should state that. DO NOT use any external knowledge.

    **Context:**
    ---
    {context}
    ---

    **Assessment Question:** {question}

    **Your JSON Answer:**
    """
'''
def get_rag_chain(retriever) -> Runnable:
    """
    Creates and returns the RAG (Retrieval-Augmented Generation) chain.
    """
    template = """
    **System:** You are an expert cybersecurity GRC auditor. Your task is to answer the assessment question based ONLY on the provided context from the uploaded document.
    - Your response MUST be in the following JSON format: {{"answer": "Clear concise short answer in one to two sentences", "reasoning": "Your brief explanation.", "evidence": "The exact quote from the context."}}

    - If the context is insufficient, your reasoning should state that. DO NOT use any external knowledge.

    **Context:**
    ---
    {context}
    ---

    **Assessment Question:** {question}

    **Your JSON Answer:**
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Using phi3:instruct as in your notebook. Ensure Ollama is running.
    llm = ChatOllama(model="phi3:instruct", format="json")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


async def get_llm_response(question: str, rag_chain: Runnable) -> Dict[str, str]:
    """
    Queries the LLM with a single question and returns the parsed JSON response.
    """
    try:
        response_str = await rag_chain.ainvoke(question)  # Use async invoke
        return json.loads(response_str)

    except json.JSONDecodeError as e:
        print(f"  - Error: LLM returned invalid JSON. Error: {e}")
        return {
            "answer": "Error",
            "reasoning": f"Failed to parse LLM response. Raw response: {response_str}",
            "evidence": "N/A",
        }
    except Exception as e:
        print(f"  - Error processing question '{question[:30]}...': {e}")
        return {
            "answer": "Error",
            "reasoning": f"Failed to get a valid response from the LLM. Error: {e}",
            "evidence": "N/A",
        }
