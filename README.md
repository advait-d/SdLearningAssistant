# System Design Learning Assistant

An advanced, context-aware web service that acts as an expert System Design interviewer and pedagogical assistant. The system answers architectural queries, reviews system designs, and dynamically routes questions between specialized Large Language Models.

Designed to be responsive, highly scalable, and structurally decoupled for continuous evolution. 

---

## ⚡️ Architecture Overview

The backend is built asynchronously on **FastAPI** to support concurrent handling of LLM generations and RAG vector searches. 

When a chat request arrives, it travels through an intelligent pipeline:
1. **Orchestrator**: Handles state, memory, and sequence validation.
2. **Intent Classification**: Evaluates the raw query using `gemini-2.5-flash` to strictly deduce the user's implicit intent (e.g., `SYSTEM_DESIGN_QUESTION`, `CONCEPT_EXPLANATION`, `DESIGN_REVIEW`).
3. **Retrieval-Augmented Generation (RAG)**: Connects to a dynamically scaled **FAISS CPU** index, vectorizing queries and safely extracting semantic context blocks matching the user's intent. Supports isolated vector indices for both OpenAI and Gemini embedding dimensionalities natively.
4. **LLM Generation**: Generates the final output using the configured provider (supports toggling seamlessly between **OpenAI gpt-4-turbo** and **Google Gemini 2.5 Flash / Pro**).
5. **Evaluation**: Synthetically analyzes the generated response using zero-shot AI evaluation before returning it to the user. Triggers a fallback loop if the response scores below confidence thresholds.

---

## 🚀 Getting Started (Local Development)

### Prerequisites
- Python 3.10+
- An OpenAI API Key (`sk-...`)
- A Google Gemini API Key (`AIza...`)

### 1. Environment Setup

Copy the example environment variables and insert your keys:
```bash
cp .env.example .env
```
*(Ensure you inject your `OPENAI_API_KEY` and `GEMINI_API_KEY` securely).*

### 2. Install Dependencies

You can run the application directly inside a lightweight virtual environment:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Application

Run the FastAPI Uvicorn ASGI server natively:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Access the swagger API documentation at `http://localhost:8000/docs`.

---

## 🐳 Docker Deployment (Production)

This service is optimized and bundled into a `python:3.10-slim` container using `gunicorn` with `uvicorn` asynchronous web workers to handle massive production traffic loads safely. 

Build the image locally:
```bash
docker build -t sd-assistant .
```

Run the container, making sure to explicitly map your `.env` file to mount API keys:
```bash
docker run -d -p 8000:10000 --env-file .env --name sd-backend sd-assistant
```
*(Note: The container implicitly binds its worker processes to `$PORT` inside the container, which defaults to `10000` via Render optimizations unless explicitly overrode).*

---

## 📡 API Usage

### `POST /api/v1/chat`
Send a stateless chat message while preserving conversation memory via `session_id`. Switch providers effortlessly by dictating the `provider`.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Design a Distributed Rate Limiter for an API gateway.",
    "session_id": "user-12345",
    "provider": "gemini" 
  }'
```

**Expected Response**:
```json
{
  "response": "To design a rate limiter, we can implement the Token Bucket algorithm backed by Redis...",
  "intent": "SYSTEM_DESIGN_QUESTION",
  "confidence_score": 0.98,
  "is_fallback": false,
  "session_id": "user-12345"
}
```
