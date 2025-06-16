#!/usr/bin/env python3
"""Start the FastAPI server for live simulation streaming."""
import uvicorn
import sys
import os

# Add backend to Python path so app module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )