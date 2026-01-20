from typing import Dict, Any
from fastapi import WebSocket

class ConnectionManager:
    """
    Manages active WebSocket connections.
    """
    def __init__(self):
        # Stores active connections mapped by client_id
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Accepts a new WebSocket connection.
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"Client {client_id} connected.")

    def disconnect(self, client_id: str):
        """
        Removes a disconnected client.
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"Client {client_id} disconnected.")

    async def send_status_update(self, client_id: str, status: str, detail: str):
        """
        Sends a JSON status update to a specific client.
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json({"status": status, "detail": detail})
            except Exception as e:
                print(f"Failed to send message to {client_id}: {e}")
                # Handle connection being closed unexpectedly
                self.disconnect(client_id)
