from fastapi import FastAPI, status, HTTPException

# to run this file fastapi application , use the command: fastapi dev main.py

from pathlib import Path

app = FastAPI()

uploads_dir = Path("./uploads")
uploads_dir.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "Hello , I am PolicyBot Auditor !"}


# api to upload evidence files
# currently it accepts client id as query parameter and creates a folder inside upload folder with client id name
@app.post("/uploadEvidence", status_code=status.HTTP_201_CREATED)
def upload_evidence(client_id: str):
    # Creates a directory for a specific client_id inside 'uploads'.
    try:
        client_upload_dir = uploads_dir / client_id
        # create directory if not exists
        client_upload_dir.mkdir(parents=True, exist_ok=True)

        return {"message": f"Evidence files uploaded for client {client_id}"}

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server permission denied. Cannot create folder",
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
