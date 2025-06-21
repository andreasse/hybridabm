"""Live run manager for real-time simulation streaming and frame caching."""
import json
import orjson
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import asyncio
from datetime import datetime

# Use same orjson options as run.py for numpy serialization
ORJSON_OPTS = (
    orjson.OPT_SERIALIZE_NUMPY
    | orjson.OPT_NAIVE_UTC
    | orjson.OPT_NON_STR_KEYS
)

def dumps(obj):
    """JSON serialization with numpy support (same as run.py)."""
    return orjson.dumps(obj, option=ORJSON_OPTS).decode('utf-8')


@dataclass
class LiveRun:
    """Represents a live simulation run with cached steps."""
    run_id: str
    start_time: datetime
    steps: List[str] = field(default_factory=list)  # JSON strings
    websocket_clients: Set[asyncio.Queue] = field(default_factory=set)
    current_step: int = 0
    is_active: bool = True


class LiveRunManager:
    """Manages live simulation runs with WebSocket streaming and step caching."""
    
    def __init__(self):
        self._runs: Dict[str, LiveRun] = {}
        self._lock = asyncio.Lock()
    
    async def create_run(self, run_id: str) -> LiveRun:
        """Create a new live run."""
        async with self._lock:
            if run_id in self._runs:
                raise ValueError(f"Run {run_id} already exists")
            
            run = LiveRun(
                run_id=run_id,
                start_time=datetime.utcnow()
            )
            self._runs[run_id] = run
            return run
    
    async def add_step(self, run_id: str, step_data: dict) -> None:
        """Add a step to the run and send minimal notification to clients."""
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")
            
            # Serialize step once and cache it (using orjson for numpy support)
            step_json = dumps(step_data)
            run.steps.append(step_json)
            run.current_step += 1
            
            # Only send notifications every 10 steps to reduce spam
            if run.current_step % 10 == 0 or run.current_step == 1:
                # Send minimal notification to WebSocket clients (not full step data)
                notification = json.dumps({
                    "type": "step_notification",
                    "t": step_data.get("t"),
                    "total_steps": len(run.steps)
                })
                
                dead_queues = set()
                for queue in run.websocket_clients:
                    try:
                        await queue.put(notification)
                    except asyncio.QueueFull:
                        # Client is too slow, skip this notification
                        pass
                    except Exception:
                        # Queue is broken, mark for removal
                        dead_queues.add(queue)
                
                # Clean up dead queues
                run.websocket_clients -= dead_queues
    
    async def get_step(self, run_id: str, step: int) -> Optional[str]:
        """Get a specific step by step number (1-indexed)."""
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            
            # Convert 1-indexed step to 0-indexed array position
            index = step - 1
            if 0 <= index < len(run.steps):
                return run.steps[index]
            return None
    
    async def add_client(self, run_id: str) -> asyncio.Queue:
        """Add a new WebSocket client to a run."""
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")
            
            # Create a queue for this client
            queue = asyncio.Queue(maxsize=100)
            run.websocket_clients.add(queue)
            return queue
    
    async def remove_client(self, run_id: str, queue: asyncio.Queue) -> None:
        """Remove a WebSocket client from a run."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run.websocket_clients.discard(queue)
    
    async def end_run(self, run_id: str) -> None:
        """Mark a run as completed."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run.is_active = False
                
                # Send final notification with completion status
                notification = json.dumps({
                    "type": "run_completed",
                    "total_steps": len(run.steps)
                })
                
                for queue in run.websocket_clients:
                    try:
                        await queue.put(notification)
                    except:
                        pass  # Don't care about errors on final notification
    
    def get_run_info(self, run_id: str) -> Optional[dict]:
        """Get basic info about a run (synchronous for convenience)."""
        run = self._runs.get(run_id)
        if not run:
            return None
        
        return {
            "run_id": run.run_id,
            "start_time": run.start_time.isoformat(),
            "total_steps": len(run.steps),
            "is_active": run.is_active,
            "connected_clients": len(run.websocket_clients)
        }
    
    def list_runs(self) -> List[dict]:
        """List all runs (synchronous for convenience)."""
        return [self.get_run_info(run_id) for run_id in self._runs.keys()]


# Global instance
live_run_manager = LiveRunManager()