# 🛡️ PolicyBot Auditor

**Automated GRC & Cybersecurity Policy Auditing using Retrieval-Augmented Generation (RAG).**

PolicyBot Auditor is a GenAI-powered tool designed to streamline the auditing process for cybersecurity and compliance frameworks (ISO 27001, SOC2, HIPAA). It allows auditors to upload raw policy documents (PDF, DOCX) and ask assessment questions. The system retrieves exact evidence from the documents and uses an LLM to generate reasoned compliance answers.

**🚀 Key Features**

* **📄 Multi-Document RAG**: Upload multiple policy documents (PDFs, Text, etc.) simultaneously to build a comprehensive knowledge base.

-- **⚡ Real-Time Progress Tracking**: Uses WebSockets to provide live feedback on parsing, chunking, and embedding processes to the frontend.

* **🔍 Evidence-Based Auditing: Every answer includes:**

  - Direct Answer: (Yes/No/Partial)

  - Reasoning: Why the policy meets/fails the criteria.

  - Evidence: Exact citations/quotes from the uploaded documents.

* **🔒 Session Isolation** : Supports multiple users concurrently. Each user session gets a dedicated, isolated Vector Store.

🧠 Local LLM Support: Optimized for privacy using local models via Ollama (e.g., Phi-3, Llama3).

🏗️ Architecture

The application follows a modern asynchronous microservices pattern:

Ingestion: User uploads documents via REST API.

Processing (Async): Background tasks parse text, split into chunks (RecursiveCharacterSplitter), and generate embeddings.

Communication: WebSockets push status updates (parsing -> chunking -> ready) to the UI.

Storage: Vectors are stored in a session-specific FAISS index.

Retrieval: When a question is asked, the system retrieves top-k relevant chunks.

Generation: An LLM (via Ollama) synthesizes the answer using the retrieved context.

🛠️ Tech Stack

Backend Framework: FastAPI (Python)

Orchestration: LangChain

Vector Database: FAISS (CPU-optimized for this build)

Embeddings: HuggingFace (all-MiniLM-L6-v2)

LLM Inference: Ollama

Containerization: Docker

⚡ Quick Start

Prerequisites

Python 3.10+

Ollama installed and running.

Model pulled: ollama pull phi3:instruct (or your preferred model).

1. Clone the Repository

git clone [https://github.com/yourusername/policybot-auditor.git](https://github.com/yourusername/policybot-auditor.git)
cd policybot-auditor


2. Set Up Virtual Environment

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Run the Server

uvicorn main:app --reload


The API will be available at http://127.0.0.1:8000.
access the Interactive Swagger Docs at http://127.0.0.1:8000/docs.

🔌 API Documentation

1. WebSocket (Real-time Status)

Endpoint: ws://localhost:8000/ws/progress/{client_id}

Connect to this before uploading files to receive progress bars/status updates.

Messages: {"status": "parsing", "detail": "..."}

2. Upload Evidence

Endpoint: POST /upload-evidences/{client_id}

Body: multipart/form-data (List of files)

Action: Starts the background ingestion process.

3. Ask Question

Endpoint: POST /ask-question/{client_id}

Body: {"question": "Does the organization have an encryption policy?"}

Response:

{
  "answer": "Yes",
  "reasoning": "The policy explicitly states that AES-256 encryption is required for data at rest.",
  "evidence": "Section 4.1: 'All sensitive data must use AES-256...'"
}


4. Reset Session

Endpoint: POST /reset-session/{client_id}

Action: Clears the vector store and uploaded files for that user.

🐳 Docker Deployment

To deploy this application as a container:

1. Build the Image

docker build -t policybot-auditor .


2. Run the Container

Note: This example assumes Ollama is running on the host machine. We use --network host for simplicity on Linux, or use host.docker.internal for Mac/Windows.

docker run -d -p 8000:8000 --name policybot policybot-auditor


📁 Project Structure

├── main.py              # FastAPI entry point & Endpoint logic
├── rag_pipeline.py      # Core RAG logic (Loading, Splitting, Embedding)
├── schemas.py           # Pydantic models for Request/Response
├── utils.py             # WebSocket connection manager
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── uploads/             # Temp storage for uploaded files (gitignored)
└── vector_store/        # Storage for FAISS indexes (gitignored)


📜 License

Distributed under the MIT License. See LICENSE for more information.

Developed for Automated Policy Compliance Auditing.
