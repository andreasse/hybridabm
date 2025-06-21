"""Event streaming endpoints for simulation events."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import List
import asyncio
import logging
import traceback

from app.application.services.event_aggregator import event_aggregator
from app.api.v1.schemas.events import SimulationEvent

# Create separate logger for WebSocket debugging
ws_logger = logging.getLogger('websocket_debug')
ws_logger.setLevel(logging.DEBUG)

# Clear existing handlers
for handler in ws_logger.handlers[:]:
    ws_logger.removeHandler(handler)

# Add file handler for WebSocket logs  
ws_handler = logging.FileHandler('websocket_debug.log', mode='w')
ws_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
ws_logger.addHandler(ws_handler)

# Also log to event_aggregator.log for consistency
event_logger = logging.getLogger('event_aggregator')
ws_logger.info("=== WEBSOCKET DEBUG LOGGER CREATED ===")
event_logger.info("=== EVENTS.PY MODULE LOADED ===")

# Log EventAggregator instance info at module load
ws_logger.info(f"WEBSOCKET MODULE: EventAggregator instance ID: {id(event_aggregator)}")
event_logger.info(f"WEBSOCKET MODULE: EventAggregator instance ID: {id(event_aggregator)}")


router = APIRouter(prefix="/events", tags=["events"])


@router.websocket("/ws/{run_id}")
async def websocket_events_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for streaming simulation events."""
    ws_logger.info(f"=== WEBSOCKET CONNECTION START === Run: {run_id}")
    event_logger.info(f"=== WEBSOCKET CONNECTION START === Run: {run_id}")
    
    try:
        await websocket.accept()
        ws_logger.info(f"WebSocket: Connection accepted for run {run_id}")
        
        # Send immediate test message to verify connection works
        await websocket.send_text('{"type": "test", "message": "connection_verified"}')
        ws_logger.info(f"WebSocket: Test message sent successfully")
        
    except Exception as e:
        ws_logger.error(f"WebSocket: Failed to accept connection or send test: {e}")
        ws_logger.error(f"WebSocket: Exception traceback: {traceback.format_exc()}")
        return
    
    queue = None
    try:
        # Add client and get initial event state
        ws_logger.info(f"WebSocket: Calling add_client for run {run_id}")
        ws_logger.info(f"WebSocket: Using EventAggregator instance ID: {id(event_aggregator)}")
        try:
            queue, init_message = await event_aggregator.add_client(run_id)
            ws_logger.info(f"WebSocket: Got init message with {len(init_message.events)} events")
        except Exception as e:
            ws_logger.error(f"WebSocket: add_client failed: {type(e).__name__}: {e}")
            ws_logger.error(f"WebSocket: add_client traceback: {traceback.format_exc()}")
            raise
        
        # Send initial event state
        ws_logger.info(f"WebSocket: Sending init message to client")
        try:
            await websocket.send_text(init_message.model_dump_json())
            ws_logger.info(f"WebSocket: Init message sent successfully")
        except Exception as e:
            ws_logger.error(f"WebSocket: Failed to send init message: {e}")
            raise
        
        # Stream event updates
        ws_logger.info(f"WebSocket: Starting update stream loop - connection should stay open")
        message_count = 0
        while True:
            try:
                ws_logger.debug(f"WebSocket: Waiting for message from queue (received {message_count} so far)")
                message_json = await queue.get()
                message_count += 1
                ws_logger.info(f"WebSocket: Received update #{message_count} from queue, sending to client")
                await websocket.send_text(message_json)
                ws_logger.info(f"WebSocket: Update #{message_count} sent successfully")
            except Exception as e:
                ws_logger.error(f"WebSocket: Error streaming event update: {type(e).__name__}: {e}")
                ws_logger.error(f"WebSocket: Stream error traceback: {traceback.format_exc()}")
                break
                
    except WebSocketDisconnect as e:
        ws_logger.info(f"WebSocket: Client disconnected normally for run {run_id}: {e}")
        event_logger.info(f"WebSocket: Client disconnected normally for run {run_id}: {e}")
    except Exception as e:
        ws_logger.error(f"WebSocket: Event WebSocket error for run {run_id}: {type(e).__name__}: {e}")
        ws_logger.error(f"WebSocket: Error traceback: {traceback.format_exc()}")
        event_logger.error(f"WebSocket: Event WebSocket error for run {run_id}: {type(e).__name__}: {e}")
    finally:
        # Clean up
        ws_logger.info(f"WebSocket: FINALLY BLOCK - Cleaning up client for run {run_id}")
        event_logger.info(f"WebSocket: FINALLY BLOCK - Cleaning up client for run {run_id}")
        if queue:
            ws_logger.info(f"WebSocket: Removing client from EventAggregator")
            await event_aggregator.remove_client(run_id, queue)
            ws_logger.info(f"WebSocket: Client removed from EventAggregator")
        else:
            ws_logger.warning(f"WebSocket: No queue to clean up for run {run_id}")
            event_logger.warning(f"WebSocket: No queue to clean up for run {run_id}")
        ws_logger.info(f"=== WEBSOCKET CONNECTION END === Run: {run_id}")
        event_logger.info(f"=== WEBSOCKET CONNECTION END === Run: {run_id}")


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