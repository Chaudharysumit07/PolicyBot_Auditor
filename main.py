from fastapi import FastAPI, status, HTTPException, UploadFile

# to run this file fastapi application , use the command: fastapi dev main.py

from pathlib import Path

import shutil


app = FastAPI()

UPLOADS_DIR = Path("./uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

ALLOWED_MIME_TYPES = ["application/pdf"]


@app.get("/")
def root():
    return {"message": "Hello , I am PolicyBot Auditor !"}


# api to upload evidence files
# currently it accepts client id as query parameter and creates a folder inside upload folder with client id name
@app.post("/uploadEvidence", status_code=status.HTTP_201_CREATED)
async def upload_evidence(client_id: str, file: UploadFile):

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            details=f"Invalid file type. Allowed: {ALLOWED_MIME_TYPES}",
        )

    # Creates a directory for a specific client_id inside 'uploads'.
    try:
        client_upload_dir = UPLOADS_DIR / client_id

        # create directory including parent directory if not exists
        client_upload_dir.mkdir(parents=True, exist_ok=True)

        # Define the final file path (Folder + Filename)
        # file.filename comes from the user's computer
        file_path = client_upload_dir / file.filename

        # --- STEP 3: SAVE THE FILE ---
        # We use 'with' (Context Manager) to ensure the file closes properly after writing
        # "wb" means "Write Binary" (essential for images/PDFs)
        with open(file_path, "wb") as file_object:

            content = await file.read()
            file_object.write(content)

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


@app.post("/getAnswer")
def get_answer():
    return {"message": "Get answer for user's question here"}


@app.delete("/cleanup_client/{client_id}")
def reset_data(client_id: str):

    target_path = UPLOADS_DIR / client_id

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
