import requests
import os
from dotenv import load_dotenv
import json
from websocket import create_connection

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


class PolicyBotAPI:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.headers = {"accept": "application/json"}

    def listen_to_progress(self):
        """
        Connects to the backend WebSocket and listens for status updates.
        Yields messages back to the UI in real-time.
        """
        # Convert http -> ws (e.g., http://localhost:8000 -> ws://localhost:8000)
        ws_url = BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
        ws_endpoint = f"{ws_url}/ws/progress/{self.client_id}"

        try:
            # 1. Open the connection
            ws = create_connection(ws_endpoint)

            ws.settimeout(10)  # Unfreeze if backend is silent for 10s

            # 2. Keep listening until the process is done
            while True:
                result = ws.recv()  # Wait for next message
                if not result:
                    break

                data = json.loads(result)
                print(f"data from websocket api.py : {data}")
                yield data  # Send data to the UI loop

                # 3. Check for exit conditions (Logic from your backend)
                status = data.get("status")
                message = data.get("message", "") or data.get("detail", "")

                # Stop listening if error or if final "System is ready" message received
                if status == "error":
                    break
                if status == "ready" and "System is ready" in message:
                    print("closing websocket")
                    break

            ws.close()

        except Exception as e:
            yield {"status": "error", "message": f"Connection error: {str(e)}"}

    def upload_documents(self, files):
        """
        Uploads a list of files to the backend.
        files: List of file-like objects from Streamlit file_uploader
        """
        url = f"{BASE_URL}/uploadEvidence/{self.client_id}"

        # Prepare files for multipart/form-data upload
        # Streamlit file objects need to be converted to a list of tuples: ('files', (filename, file_bytes, content_type))
        files_payload = []
        for file in files:
            files_payload.append(("files", (file.name, file.getvalue(), file.type)))

        try:
            response = requests.post(url, files=files_payload)
            response.raise_for_status()  # Raise error for 4xx or 5xx
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def ask_question(self, question: str):
        """
        Sends a question to the RAG backend.
        """
        url = f"{BASE_URL}/ask-question/{self.client_id}"
        payload = {"question": question}

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            # Try to get the detail message from backend error
            try:
                error_detail = response.json().get("detail", str(http_err))
            except:
                error_detail = str(http_err)
            return {"error": error_detail}
        except Exception as e:
            return {"error": str(e)}

    def reset_session(self):
        """
        Clears the backend session data.
        """
        url = f"{BASE_URL}/cleanup_client/{self.client_id}"
        try:
            response = requests.delete(url)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
