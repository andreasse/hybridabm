"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import live_runs, events
from app.application.services import simulation_service as _ss

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
app.include_router(events.router, prefix="/api/v1")

@app.on_event("startup")
async def _startup_mp():
    """Parent process builds the pool/manager exactly once."""
    _ss.init_mp()

@app.on_event("shutdown")
async def _close_mp_manager():
    """Tear down the multiprocessing machinery cleanly."""
    # 0) stop any simulations that are still streaming
    await _ss.simulation_service.shutdown()

    # 1) kill the worker pool (now idle) fast
    if _ss._EXECUTOR:
        _ss._EXECUTOR.shutdown(wait=False, cancel_futures=True)

    # 2) close the Manager's listener socket
    if _ss._MANAGER:
        _ss._MANAGER.shutdown()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "hybrid-threat-simulation"}