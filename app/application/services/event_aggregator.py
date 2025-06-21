"""Simple event table for tracking and streaming simulation events."""
import asyncio
from typing import Dict, List, Set
from dataclasses import dataclass, field
import json
import logging
import os

# Set up logging for event aggregator
logger = logging.getLogger('event_aggregator')
logger.setLevel(logging.DEBUG)
# Use append mode and add separators to identify different processes
handler = logging.FileHandler('event_aggregator.log', mode='a')
handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
logger.addHandler(handler)
logger.info("="*80)
logger.info(f"EventAggregator module loaded - PID: {os.getpid()}")
logger.info("="*80)

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
        self._async_lock = asyncio.Lock()
        self._instance_id = id(self)
        import traceback
        logger.info(f"EventAggregator: INSTANCE CREATED - ID {self._instance_id}")
        logger.info(f"EventAggregator: Creation stack trace: {traceback.format_stack()}")
    
    async def create_run(self, run_id: str) -> RunEventTable:
        """Initialize event table for a new run."""
        async with self._async_lock:
            if run_id in self._runs:
                return self._runs[run_id]
            
            table = RunEventTable(run_id=run_id)
            self._runs[run_id] = table
            return table
    
    async def add_event(self, run_id: str, event: SimulationEvent) -> None:
        """Add a single event to the table and broadcast update."""
        logger.info(f"EventAggregator: Received event {event.type} at t={event.t} for run {run_id}")
        logger.info(f"EventAggregator: INSTANCE ID {self._instance_id}")
        logger.info(f"EventAggregator: Current tables: {list(self._runs.keys())}")
        async with self._async_lock:
            table = self._runs.get(run_id)
            if not table:
                logger.info(f"EventAggregator: Creating new table for event {event.type}")
                # Create table directly without calling create_run to avoid deadlock
                table = RunEventTable(run_id=run_id)
                self._runs[run_id] = table
            else:
                logger.info(f"EventAggregator: Using existing table with {len(table.websocket_clients)} clients")
            
            table.events.append(event)
            logger.info(f"EventAggregator: Event added to table, now have {len(table.events)} total events")
            await self._broadcast_update(run_id, [event])
    
    async def add_events(self, run_id: str, events: List[SimulationEvent]) -> None:
        """Add multiple events to the table and broadcast update."""
        if not events:
            return
            
        async with self._async_lock:
            table = self._runs.get(run_id)
            if not table:
                # Create table directly without calling create_run to avoid deadlock
                table = RunEventTable(run_id=run_id)
                self._runs[run_id] = table
            
            table.events.extend(events)
            await self._broadcast_update(run_id, events)
    
    async def add_client(self, run_id: str) -> tuple[asyncio.Queue, EventInitMessage]:
        """Add WebSocket client and return queue + initial state."""
        logger.info(f"EventAggregator: WebSocket client connecting for run {run_id}")
        logger.info(f"EventAggregator: INSTANCE ID {self._instance_id}")
        logger.info(f"EventAggregator: Current tables: {list(self._runs.keys())}")
        async with self._async_lock:
            table = self._runs.get(run_id)
            if not table:
                logger.info(f"EventAggregator: Creating new table for client connection")
                # Create table directly without calling create_run to avoid deadlock
                table = RunEventTable(run_id=run_id)
                self._runs[run_id] = table
            else:
                logger.info(f"EventAggregator: Using existing table with {len(table.events)} events")
            
            queue = asyncio.Queue(maxsize=100)
            table.websocket_clients.add(queue)
            logger.info(f"EventAggregator: Client added, now have {len(table.websocket_clients)} clients")
            
            init_message = EventInitMessage(
                events=table.events.copy(),
                provider_down_steps=table.get_provider_down_steps(),
                summary=table.get_summary()
            )
            logger.info(f"EventAggregator: Sending init message with {len(table.events)} events")
            
            return queue, init_message
    
    async def remove_client(self, run_id: str, queue: asyncio.Queue) -> None:
        """Remove WebSocket client."""
        async with self._async_lock:
            table = self._runs.get(run_id)
            if table:
                table.websocket_clients.discard(queue)
                logger.info(f"EventAggregator: Client removed, now have {len(table.websocket_clients)} clients")
    
    def get_run_events(self, run_id: str) -> List[SimulationEvent] | None:
        """Get all events for a run (synchronous for REST API)."""
        table = self._runs.get(run_id)
        return table.events.copy() if table else None
    
    async def _broadcast_update(self, run_id: str, new_events: List[SimulationEvent]) -> None:
        """Broadcast event update to all connected clients."""
        table = self._runs.get(run_id)
        if not table:
            logger.warning(f"EventAggregator: No table found for run {run_id}")
            return
        if not table.websocket_clients:
            logger.warning(f"EventAggregator: No WebSocket clients for run {run_id}")
            return
        
        logger.info(f"EventAggregator: Broadcasting {len(new_events)} new events to {len(table.websocket_clients)} clients")
        
        update_message = EventUpdateMessage(
            events=table.events.copy(),
            provider_down_steps=table.get_provider_down_steps(),
            new_events=new_events,
            summary=table.get_summary()
        )
        
        message_json = update_message.model_dump_json()
        dead_queues = set()
        
        for queue in table.websocket_clients:
            try:
                await queue.put(message_json)
                logger.debug(f"EventAggregator: Message sent to client successfully")
            except asyncio.QueueFull:
                logger.warning(f"EventAggregator: Client queue full, message dropped")
            except Exception as e:
                logger.error(f"EventAggregator: Failed to send message to client: {e}")
                dead_queues.add(queue)
        
        if dead_queues:
            logger.info(f"EventAggregator: Removing {len(dead_queues)} dead clients")
            table.websocket_clients -= dead_queues


# Module-level singleton pattern
_event_aggregator_instance = None

def get_event_aggregator():
    """Get the singleton EventAggregator instance."""
    global _event_aggregator_instance
    if _event_aggregator_instance is None:
        logger.info("EventAggregator: Creating module-level singleton instance")
        _event_aggregator_instance = EventAggregator()
        logger.info(f"EventAggregator: Module-level singleton created - ID {id(_event_aggregator_instance)}")
    else:
        logger.info(f"EventAggregator: Reusing module-level singleton - ID {id(_event_aggregator_instance)}")
    return _event_aggregator_instance

# Global instance using proper singleton pattern
event_aggregator = get_event_aggregator()