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
from temp import rag_pipeline, schemas, utils

from typing import List, Dict, Any

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PolicyBot Auditor API")


# CORS (Allowing frontend access)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


UPLOADS_DIR = Path("./uploads")
VECTOR_STORE_DIR = Path("./vector_store")

UPLOADS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)


# In memory database to hold the the RAG resource for each client session
# app_state["client_id"]={"chain":rag_chain,"db_path":"..."}

app_state: Dict[str, Dict[str, Any]] = {}

ALLOWED_MIME_TYPES = ["application/pdf"]

manager = utils.ConnectionManager()


async def process_and_store_evidences(client_id: str, file_paths: List[Path]):

    db_path = ""

    try:
        db_path = VECTOR_STORE_DIR / client_id

        retriever = await rag_pipeline.create_vector_store(
            file_paths=file_paths,
            db_path=db_path,
            client_id=client_id,
            websocket_manager=manager,
        )

        await manager.send_status_update(client_id, "ready", "Creating RAG chain..")
        rag_chain = rag_pipeline.get_rag_chain(retriever)

        app_state[client_id] = {"chain": rag_chain, "db_path": db_path}

        await manager.send_status_update(
            client_id, "ready", "System is ready. You can now ask questions."
        )

    except Exception as e:
        print(f"Error processing for {client_id}:{str(e)}")
        await manager.send_status_update(
            client_id, "error", f"An error occured : {str(e)}"
        )

        # clean up failed build
        if db_path and db_path.exists():
            shutil.rmtree(db_path)


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
    session = app_state.get(client_id)
    if not session or "chain" not in session:
        raise HTTPException(
            status_code=404,
            detail="No document session found for this client. Please upload documents first.",
        )

    rag_chain = session["chain"]

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

    session = app_state.pop(client_id, None)
    # Delete the on-disk vector store
    if session:
        db_path = session.get("db_path")

        if db_path and db_path.exists():
            shutil.rmtree(db_path)

    target_path = UPLOADS_DIR / client_id

    # Delete any lingering uploads
    if not target_path.exists():

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for {client_id} client",
        )

    try:
        if target_path.is_file():

            target_path.unlink(missing_ok=True)

        elif target_path.is_dir():

            shutil.rmtree(target_path)

        return {"message": f"Data successfully deleted for the client {client_id}"}

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Permission Denied .Server cannot delete this file.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting file: {str(e)}",
        )
