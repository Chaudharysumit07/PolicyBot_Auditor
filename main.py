from fastapi import FastAPI, status, HTTPException, UploadFile

# to run this file fastapi application , use the command: fastapi dev main.py

from pathlib import Path

app = FastAPI()

uploads_dir = Path("./uploads")
uploads_dir.mkdir(exist_ok=True)

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
        client_upload_dir = uploads_dir / client_id

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
