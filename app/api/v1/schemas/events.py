"""Schemas for event streaming WebSocket messages."""
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel


class SimulationEvent(BaseModel):
    """Individual simulation event."""
    t: int  # Timestep (1-indexed to match frontend)
    type: Literal["CYBER", "MISINFO", "COMBO", "PROVIDER_DOWN"]
    provider: Optional[int] = None  # Provider ID for PROVIDER_DOWN events
    
    class Config:
        schema_extra = {
            "example": {
                "t": 150,
                "type": "CYBER",
                "provider": None
            }
        }


class EventSummary(BaseModel):
    """Summary counts of events by type."""
    total_cyber: int = 0
    total_misinfo: int = 0
    total_combo: int = 0
    total_provider_down: int = 0


class EventStreamMessage(BaseModel):
    """Base class for all event stream messages."""
    type: str


class EventInitMessage(EventStreamMessage):
    """Initial message sent when client connects."""
    type: Literal["event_init"] = "event_init"
    events: List[SimulationEvent]
    provider_down_steps: List[int]  # For backward compatibility with frontend
    summary: EventSummary
    
    class Config:
        schema_extra = {
            "example": {
                "type": "event_init",
                "events": [
                    {"t": 150, "type": "CYBER", "provider": None},
                    {"t": 250, "type": "PROVIDER_DOWN", "provider": 2}
                ],
                "provider_down_steps": [250],
                "summary": {
                    "total_cyber": 1,
                    "total_misinfo": 0,
                    "total_combo": 0,
                    "total_provider_down": 1
                }
            }
        }


class EventUpdateMessage(EventStreamMessage):
    """Incremental update when new events occur."""
    type: Literal["event_update"] = "event_update"
    events: List[SimulationEvent]  # All events (not just new ones)
    provider_down_steps: List[int]  # All provider down steps
    new_events: List[SimulationEvent]  # Just the new events since last update
    summary: EventSummary
    
    class Config:
        schema_extra = {
            "example": {
                "type": "event_update",
                "events": [
                    {"t": 150, "type": "CYBER", "provider": None},
                    {"t": 250, "type": "PROVIDER_DOWN", "provider": 2},
                    {"t": 350, "type": "MISINFO", "provider": None}
                ],
                "provider_down_steps": [250],
                "new_events": [
                    {"t": 350, "type": "MISINFO", "provider": None}
                ],
                "summary": {
                    "total_cyber": 1,
                    "total_misinfo": 1,
                    "total_combo": 0,
                    "total_provider_down": 1
                }
            }
        }


class EventErrorMessage(EventStreamMessage):
    """Error message for event stream."""
    type: Literal["event_error"] = "event_error"
    error: str
    code: Optional[str] = None