"""Live run endpoints for WebSocket streaming and frame access."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional, Dict, Any
import asyncio
import json
import subprocess
import uuid
from datetime import datetime
from pydantic import BaseModel

from app.application.services.live_run_manager import live_run_manager


class SimulateResponse(BaseModel):
    runId: str
    message: str

router = APIRouter(prefix="/runs", tags=["live-runs"])


@router.post("/simulate", response_model=SimulateResponse)
async def start_simulation(params: Dict[str, Any]):
    """Start a new simulation with given parameters."""
    # For now, ignore params and start run.py - we'll add parameter passing later
    try:
        # Start the simulation process in the background
        process = subprocess.Popen(
            ["python", "run.py"],
            cwd="/home/andreas/hybrid_article/backend",
            stdout=None,  # Inherit our stdout
            stderr=None,  # Inherit our stderr
            start_new_session=True,
            text=True
        )
        
        # Read the output to get the actual run ID
        run_id = None
        timeout = 10  # 10 second timeout
        
        for _ in range(timeout * 10):  # Check every 100ms
            if process.poll() is not None:
                # Process finished (probably with error)
                stdout, stderr = process.communicate()
                raise HTTPException(
                    status_code=500,
                    detail=f"Simulation process exited early. Stderr: {stderr}"
                )
            
            # Check if run ID is available in live_run_manager
            runs = live_run_manager.list_runs()
            if runs:
                # Get the most recent run
                latest_run = max(runs, key=lambda x: x.get('start_time', ''))
                run_id = latest_run['run_id']
                break
                
            await asyncio.sleep(0.1)  # Wait 100ms
        
        if not run_id:
            # Fallback: generate placeholder run ID
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_id = f"exp3_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        return SimulateResponse(
            runId=run_id,
            message=f"Simulation started with process ID {process.pid}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start simulation: {str(e)}"
        )


@router.post("/{run_id}/register")
async def register_run(run_id: str):
    """Register a new run that will start streaming."""
    try:
        await live_run_manager.create_run(run_id)
        return {"status": "registered", "run_id": run_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{run_id}/step")
async def add_step(run_id: str, step_data: Dict[str, Any]):
    """Add a step to the run."""
    try:
        await live_run_manager.add_step(run_id, step_data)
        step_t = step_data.get('t', 'unknown')
        if step_t <= 5:  # Only log first few steps
            print(f"Received and broadcasted step {step_t} for run {run_id}")
        return {"status": "ok"}
    except ValueError as e:
        print(f"Error adding step for run {run_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{run_id}/end")
async def end_run(run_id: str):
    """Mark a run as completed."""
    try:
        await live_run_manager.end_run(run_id)
        return {"status": "ended"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for chunked frame streaming."""
    await websocket.accept()
    
    queue = None
    try:
        # Register this client with the run
        queue = await live_run_manager.add_client(run_id)
        
        # Handle live step notifications (minimal data)
        async def handle_live_updates():
            """Handle live step notifications from simulation"""
            while True:
                try:
                    notification_json = await queue.get()
                    # Forward the notification directly (it's already formatted)
                    await websocket.send_text(notification_json)
                except Exception as e:
                    print(f"Error handling live notification: {e}")
                    break
        
        async def handle_chunk_requests():
            """Handle chunk requests from frontend"""
            while True:
                try:
                    message = await websocket.receive_text()
                    request = json.loads(message)
                    
                    if request.get("action") == "requestFrames":
                        start_step = request.get("start", 1)
                        end_step = request.get("end", 10)
                        request_id = request.get("requestId", f"{start_step}-{end_step}")
                        
                        print(f"[WebSocket] Handling chunk request: {start_step}-{end_step} (id: {request_id})")
                        
                        # Collect frames for the requested range
                        frames = []
                        for step in range(start_step, end_step + 1):
                            step_json = await live_run_manager.get_step(run_id, step)
                            if step_json:
                                frames.append(json.loads(step_json))
                        
                        # Send chunk response
                        response = {
                            "type": "chunk_response",
                            "frames": frames,
                            "start": start_step,
                            "end": end_step,
                            "requestId": request_id
                        }
                        await websocket.send_text(json.dumps(response))
                        
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error handling chunk request: {e}")
                    break
        
        # Run both handlers concurrently
        await asyncio.gather(
            handle_live_updates(),
            handle_chunk_requests()
        )
            
    except WebSocketDisconnect:
        # Client disconnected normally
        pass
    except Exception as e:
        # Log error in production
        print(f"WebSocket error for run {run_id}: {e}")
    finally:
        # Clean up
        if queue:
            await live_run_manager.remove_client(run_id, queue)


@router.get("/{run_id}/steps/{step}")
async def get_step(run_id: str, step: int):
    """Get a specific step by step number (1-indexed)."""
    step_json = await live_run_manager.get_step(run_id, step)
    
    if step_json is None:
        raise HTTPException(
            status_code=404,
            detail=f"Step not found for run {run_id} step {step}"
        )
    
    # Parse and return as JSON (not string)
    return json.loads(step_json)


@router.get("/{run_id}/info")
async def get_run_info(run_id: str):
    """Get information about a run."""
    info = live_run_manager.get_run_info(run_id)
    
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found"
        )
    
    return info


@router.get("/")
async def list_runs():
    """List all available runs."""
    return {
        "runs": live_run_manager.list_runs()
    }