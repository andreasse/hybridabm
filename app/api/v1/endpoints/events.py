"""Event streaming endpoints for simulation events."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import List
import asyncio

from app.application.services.event_aggregator import event_aggregator
from app.api.v1.schemas.events import SimulationEvent


router = APIRouter(prefix="/events", tags=["events"])


@router.websocket("/ws/{run_id}")
async def websocket_events_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for streaming simulation events."""
    await websocket.accept()
    
    queue = None
    try:
        # Add client and get initial event state
        queue, init_message = await event_aggregator.add_client(run_id)
        
        # Send initial event state
        await websocket.send_text(init_message.json())
        
        # Stream event updates
        while True:
            try:
                message_json = await queue.get()
                await websocket.send_text(message_json)
            except Exception as e:
                print(f"Error streaming event update: {e}")
                break
                
    except WebSocketDisconnect:
        # Client disconnected normally
        pass
    except Exception as e:
        print(f"Event WebSocket error for run {run_id}: {e}")
    finally:
        # Clean up
        if queue:
            await event_aggregator.remove_client(run_id, queue)


@router.post("/{run_id}/add")
async def add_event(run_id: str, event: SimulationEvent):
    """Add a single event to the run (called by simulation)."""
    try:
        await event_aggregator.add_event(run_id, event)
        return {"status": "ok", "event": event}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{run_id}/add-batch")
async def add_events(run_id: str, events: List[SimulationEvent]):
    """Add multiple events to the run (called by simulation)."""
    try:
        await event_aggregator.add_events(run_id, events)
        return {"status": "ok", "count": len(events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}")
async def get_events(run_id: str):
    """Get all events for a run (for debugging)."""
    events = event_aggregator.get_run_events(run_id)
    if events is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"events": events}