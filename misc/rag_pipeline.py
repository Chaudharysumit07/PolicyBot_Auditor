import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Any

from operator import itemgetter

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


# Internal import
from misc import utils


#for chroma db implementation
import chromadb
from langchain_chroma import Chroma


# --- Configuration ---
# Use a global variable to cache the embeddings model
EMBEDDINGS_MODEL = None

OLLAMA_BASE_URL= "http://host.docker.internal:11434"


# Initialize Chroma Client (shared across all workers)
# The host matches the service name in docker-compose.yml
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "chroma-db"),
    port=8000
)


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
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, collection_name=client_id, client= chroma_client)

    
    await websocket_manager.send_status_update(
        client_id, "embedding", "✅ Documents indexed in central Vector DB.")

    # 5. Create and return the retriever
    return vectorstore.as_retriever(search_kwargs={"k": 3})


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
def get_chroma_retriever(client_id: str):
    """
    Connects to the central ChromaDB server and retrieves the collection for a specific client.
    This is called by every worker on-demand to remain 'Stateless'.
    """
    embeddings = get_embeddings_model()
    
    # Check if collection exists
    try:
        chroma_client.get_collection(name=client_id)
    except Exception:
        return None

    vector_store = Chroma(
        client=chroma_client,
        collection_name=client_id,
        embedding_function=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})

def get_rag_chain(retriever) -> Runnable:
    """
    Creates and returns the RAG (Retrieval-Augmented Generation) chain.
    """
    template = """
    You are an expert Cybersecurity Auditor. Answer the question based ONLY on the provided context.

    **Strict Rules:**
    1. If the text mentions a specific Section Number (e.g., "4.1", "Appendix B", "Section IX"), you MUST cite it.
    2. If the answer is not in the context, say "Evidence not found."

    **Output Format (JSON):**
    {{
        "answer": "One line short precise answer",
        "reasoning": "Explanation based on the text",
        "clause_reference": "The specific section number (e.g., 'Section 5.3.1') or 'N/A'",
        "quote": "The exact sentence from the text supporting the answer"
    }}

    Context:
    {context}

    Question: {question}
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Using phi3:instruct as in your notebook. Ensure Ollama is running.
    llm = ChatOllama(model="phi3:instruct", format="json",base_url = OLLAMA_BASE_URL)

    # 1. Setup Retrieval: Get context and keep the question
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    # 2. Setup Generation: Feed context+question into prompt -> llm -> parser
    generation_pipeline = prompt | llm | StrOutputParser()

    # 3. Combine: Run retrieval, then run generation AND pass through the original docs
    rag_chain = setup_and_retrieval | RunnableParallel(
        {"result": generation_pipeline, "source_documents": itemgetter("context")}
    )

    return rag_chain





async def get_llm_response(question: str, rag_chain) -> Dict[str, Any]:
    """
    Queries the LLM and extracts Answer + Page Numbers (Metadata) + Clause (JSON).
    """
    try:
        # 1. Invoke the Chain
        # This returns a DICT: {'result': "...", 'source_documents': [...]}
        response_payload = await rag_chain.ainvoke(question)

        # 2. Extract the raw LLM text and the Source Documents
        llm_text_output = response_payload.get("result", "")
        source_docs = response_payload.get("source_documents", [])

        # 3. Clean and Parse the JSON from the LLM text
        # (LLMs often wrap JSON in ```json markers, so we remove them)
        clean_json = llm_text_output.strip().replace("```json", "").replace("```", "")
        parsed_output = json.loads(clean_json)

        # 4. Extract Page Numbers from Metadata
        # We look at every chunk used and get its page number (usually 0-indexed)
        unique_pages = sorted(
            list(
                set([f"Page {doc.metadata.get('page', 0) + 1}" for doc in source_docs])
            )
        )

        # Extract File Names
        unique_files = sorted(
            list(
                set(
                    [
                        Path(doc.metadata.get("source", "Unknown")).name
                        for doc in source_docs
                    ]
                )
            )
        )

        # 5. Return the Combined Data
        return {
            "answer": parsed_output.get("answer", "N/A"),
            "reasoning": parsed_output.get("reasoning", "N/A"),
            "evidence": parsed_output.get("quote", "N/A"),
            # We combine the LLM's "Clause" with the PDF's "Page Number"
            "sources": [
                f"Ref: {parsed_output.get('clause_reference', 'N/A')}",
                f"Location: {', '.join(unique_pages)}",
                f"Files: {', '.join(unique_files)}",
            ],
        }

    except json.JSONDecodeError:
        print(f"JSON Error. Raw output: {llm_text_output}")
        return {
            "answer": "Error parsing AI response",
            "reasoning": "The AI model did not return valid JSON.",
            "evidence": llm_text_output,  # Return raw text so you can see what went wrong
            "sources": [],
        }

    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "answer": "System Error",
            "reasoning": str(e),
            "evidence": "N/A",
            "sources": [],
        }
