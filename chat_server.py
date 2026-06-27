"""Compatibility entrypoint for the FastAPI app.

Preferred command:
    uvicorn ai_server.api.main:app --reload --port 8000
"""

from ai_server.api.main import app
