"""Simple event table for tracking and streaming simulation events."""
import asyncio
from typing import Dict, List, Set
from dataclasses import dataclass, field
import json

from app.api.v1.schemas.events import (
    SimulationEvent, 
    EventSummary,
    EventInitMessage,
    EventUpdateMessage
)


@dataclass
class RunEventTable:
    """Simple table of events for a run."""
    run_id: str
    events: List[SimulationEvent] = field(default_factory=list)
    websocket_clients: Set[asyncio.Queue] = field(default_factory=set)
    
    def get_provider_down_steps(self) -> List[int]:
        """Extract provider down steps for backward compatibility."""
        return [e.t for e in self.events if e.type == "PROVIDER_DOWN"]
    
    def get_summary(self) -> EventSummary:
        """Calculate event summary counts."""
        summary = EventSummary()
        for event in self.events:
            if event.type == "CYBER":
                summary.total_cyber += 1
            elif event.type == "MISINFO":
                summary.total_misinfo += 1
            elif event.type == "COMBO":
                summary.total_combo += 1
            elif event.type == "PROVIDER_DOWN":
                summary.total_provider_down += 1
        return summary


class EventAggregator:
    """Dumb event table that receives events and streams them to clients."""
    
    def __init__(self):
        self._runs: Dict[str, RunEventTable] = {}
        self._lock = asyncio.Lock()
    
    async def create_run(self, run_id: str) -> RunEventTable:
        """Initialize event table for a new run."""
        async with self._lock:
            if run_id in self._runs:
                return self._runs[run_id]
            
            table = RunEventTable(run_id=run_id)
            self._runs[run_id] = table
            return table
    
    async def add_event(self, run_id: str, event: SimulationEvent) -> None:
        """Add a single event to the table and broadcast update."""
        async with self._lock:
            table = self._runs.get(run_id)
            if not table:
                table = await self.create_run(run_id)
            
            table.events.append(event)
            await self._broadcast_update(run_id, [event])
    
    async def add_events(self, run_id: str, events: List[SimulationEvent]) -> None:
        """Add multiple events to the table and broadcast update."""
        if not events:
            return
            
        async with self._lock:
            table = self._runs.get(run_id)
            if not table:
                table = await self.create_run(run_id)
            
            table.events.extend(events)
            await self._broadcast_update(run_id, events)
    
    async def add_client(self, run_id: str) -> tuple[asyncio.Queue, EventInitMessage]:
        """Add WebSocket client and return queue + initial state."""
        async with self._lock:
            table = self._runs.get(run_id)
            if not table:
                table = await self.create_run(run_id)
            
            queue = asyncio.Queue(maxsize=100)
            table.websocket_clients.add(queue)
            
            init_message = EventInitMessage(
                events=table.events.copy(),
                provider_down_steps=table.get_provider_down_steps(),
                summary=table.get_summary()
            )
            
            return queue, init_message
    
    async def remove_client(self, run_id: str, queue: asyncio.Queue) -> None:
        """Remove WebSocket client."""
        async with self._lock:
            table = self._runs.get(run_id)
            if table:
                table.websocket_clients.discard(queue)
    
    async def _broadcast_update(self, run_id: str, new_events: List[SimulationEvent]) -> None:
        """Broadcast event update to all connected clients."""
        table = self._runs.get(run_id)
        if not table or not table.websocket_clients:
            return
        
        update_message = EventUpdateMessage(
            events=table.events.copy(),
            provider_down_steps=table.get_provider_down_steps(),
            new_events=new_events,
            summary=table.get_summary()
        )
        
        message_json = update_message.json()
        dead_queues = set()
        
        for queue in table.websocket_clients:
            try:
                await queue.put(message_json)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead_queues.add(queue)
        
        table.websocket_clients -= dead_queues


# Global instance
event_aggregator = EventAggregator()