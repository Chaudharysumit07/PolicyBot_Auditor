from pydantic import BaseModel
from typing import List


class QuestionRequest(BaseModel):
    """
    The shape of the request body for /ask-question
    """

    question: str


class AnswerResponse(BaseModel):
    """
    The shape of the JSON response from /ask-question
    """

    answer: str
    reasoning: str
    evidence: str
    sources: List[str]


class StatusResponse(BaseModel):
    """
    The shape of the JSON message sent over WebSocket
    """

    status: str  # e.g., "parsing", "chunking", "embedding", "ready", "error"
    detail: str  # e.g., "Parsing 5 files...", "✅ System is ready."
