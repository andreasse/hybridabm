#!/usr/bin/env python3
"""Start the FastAPI server for live simulation streaming."""
import uvicorn
import sys
import os
import logging

# Add backend to Python path so app module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Custom logging filter to reduce noise from step endpoint
class StepEndpointFilter(logging.Filter):
    def filter(self, record):
        # Filter out step endpoint access logs
        if hasattr(record, 'args') and record.args:
            message = record.getMessage()
            if "/step HTTP/" in message and "200 OK" in message:
                return False
        return True

if __name__ == "__main__":
    # Configure logging to filter out noisy step endpoint logs
    logging.getLogger("uvicorn.access").addFilter(StepEndpointFilter())
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )