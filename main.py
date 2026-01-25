from fastapi import (
    FastAPI,
    status,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
)

# to run this file fastapi application , use the command: fastapi dev main.py

from pathlib import Path

import shutil
from misc import rag_pipeline, schemas, utils

from typing import List, Dict, Any

from fastapi.middleware.cors import CORSMiddleware

import asyncio

import chromadb
import os

app = FastAPI(title="PolicyBot Auditor API")


# CORS (Allowing frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOADS_DIR = Path("./uploads")
# VECTOR_STORE_DIR = Path("./vector_store")

UPLOADS_DIR.mkdir(exist_ok=True)
# VECTOR_STORE_DIR.mkdir(exist_ok=True)


chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "chroma-db"), 
    port=8000
)

ALLOWED_MIME_TYPES = ["application/pdf"]

manager = utils.ConnectionManager()


async def process_and_store_evidences(client_id: str, file_paths: List[Path]):

    await  asyncio.sleep(2)

    # db_path = ""

    try:
        # db_path = VECTOR_STORE_DIR / client_id

        retriever = await rag_pipeline.create_vector_store(
            file_paths=file_paths,
            client_id=client_id,
            websocket_manager=manager,
        )


        # app_state[client_id] = {"chain": rag_chain}

        await manager.send_status_update(
            client_id, "ready", "System is ready. You can now ask questions."
        )
        print(f"System is ready. You can now ask questions.")

    except Exception as e:
        print(f"Error processing for {client_id}:{str(e)}")
        await manager.send_status_update(
            client_id, "error", f"An error occured : {str(e)}"
        )

        # # clean up failed build
        # if db_path and db_path.exists():
        #     shutil.rmtree(db_path)


# ------------websocket_endpoint--------------------


@app.websocket("/ws/progress/{client_id}")
async def websocket_progress_endpoint(websocket: WebSocket, client_id: str):
    """
    Handles the WebSocket connection for a client.
    Listens for messages and broadcasts status updates.
    """
    await manager.connect(websocket, client_id)
    try:
        while True:
            # The WebSocket just stays open, listening.
            data = await websocket.receive_text()
            print(f"Recieved message from {client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client {client_id} disconnected.")


# ---------------------------------------------------


# ----------------API Endpoints----------------------


@app.get("/")
def root():
    return {"message": "Hello , I am PolicyBot Auditor !"}


# api to upload evidence files
# currently it accepts client id as query parameter and creates a folder inside upload folder with client id name
@app.post("/uploadEvidence/{client_id}", status_code=status.HTTP_201_CREATED)
async def upload_evidence(
    background_tasks: BackgroundTasks, client_id: str, files: List[UploadFile]
):

    # Creates a directory for a specific client_id inside 'uploads'.
    try:
        client_upload_dir = UPLOADS_DIR / client_id

        # create directory including parent directory if not exists
        client_upload_dir.mkdir(parents=True, exist_ok=True)

        file_paths = []

        for file in files:

            if file.content_type not in ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    details=f"Invalid file type. Allowed: {ALLOWED_MIME_TYPES}",
                )

            # Define the final file path (Folder + Filename)
            # file.filename comes from the user's computer
            file_path = client_upload_dir / file.filename

            # --- STEP 3: SAVE THE FILE ---
            # We use 'with' (Context Manager) to ensure the file closes properly after writing
            # "wb" means "Write Binary" (essential for images/PDFs)
            with open(file_path, "wb") as file_object:

                content = await file.read()
                file_object.write(content)

            file_paths.append(file_path)

        background_tasks.add_task(process_and_store_evidences, client_id, file_paths)

        return {"message": f"{file.filename} uploaded successfully"}

    except PermissionError:
        # if script don't have permission to create directory at given location
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server permission denied. Cannot create folder or cannot save file",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured:{str(e)}",
        )


# api to get answer for user's question--> LLM processing inside this api


@app.post("/ask-question/{client_id}", response_model=schemas.AnswerResponse)
async def ask_question(request: schemas.QuestionRequest, client_id: str):
    """
    Asks a question against the vector store created for the client_id.
    """
    # 1. Find the RAG chain for this client
    # session = app_state.get(client_id)

    retriever = rag_pipeline.get_chroma_retriever(client_id)  

    if not retriever:
        raise HTTPException(status_code=404, detail="Session not found")

    
    await manager.send_status_update(client_id, "ready", "Creating RAG chain..")
    rag_chain = rag_pipeline.get_rag_chain(retriever)

    # 2. Get the response from the pipeline
    try:
        response_dict = await rag_pipeline.get_llm_response(
            question=request.question, rag_chain=rag_chain
        )
        return schemas.AnswerResponse(**response_dict)

    except Exception as e:
        print(f"Error during RAG chain invocation for {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {e}")


@app.delete("/cleanup_client/{client_id}")
def reset_data(client_id: str):

    """
    Performs a full cleanup for a specific client:
    1. Drops the isolated collection from the central ChromaDB server.
    2. Deletes the uploaded PDFs from the persistent Docker volume.
    """

    # --- 1. CHROMA DB CLEANUP ---
    # We tell the central database server to delete the 'Locked Room' (Collection)
    try:
        # Check if collection exists before attempting delete to avoid unnecessary errors
        collections = [c.name for c in chroma_client.list_collections()]
        if client_id in collections:
            chroma_client.delete_collection(name=client_id)
            print(f"ChromaDB collection for {client_id} deleted.")
    except Exception as e:
        # We don't raise an error here yet, as the files might still need deleting
        print(f"Warning: Could not delete Chroma collection: {str(e)}")
    
    # --- 2. VOLUME STORAGE CLEANUP ---
    # This path is mapped to your laptop via Docker Volumes
    target_path = UPLOADS_DIR / client_id

    if not target_path.exists():
        # If neither the DB nor the files exist, then there's nothing to do
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data or session found for client {client_id}",
        )

    try:
        # Standard recursive delete for the client's folder in the volume
        if target_path.is_dir():
            shutil.rmtree(target_path)
        elif target_path.is_file():
            target_path.unlink()

        return {
            "status": "success",
            "message": f"All data for client {client_id} has been wiped from Database and Volume."
        }

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Permission Denied: Docker cannot modify the volume files. Check folder permissions on host.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing disk cleanup: {str(e)}",
        )