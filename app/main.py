"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import live_runs

# Create FastAPI instance
app = FastAPI(
    title="Hybrid Threat Simulation API",
    version="1.0.0",
    description="Real-time simulation streaming and control API"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(live_runs.router, prefix="/api/v1")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "hybrid-threat-simulation"}