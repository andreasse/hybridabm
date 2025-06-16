"""Integration with the existing simulation code."""
import asyncio
from typing import Optional
from datetime import datetime
import uuid

from app.application.services.live_run_manager import live_run_manager


class SimulationBridge:
    """Bridge between the existing simulation and the new streaming infrastructure."""
    
    def __init__(self):
        self.current_run_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def start_run(self) -> str:
        """Start a new simulation run."""
        # Generate unique run ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.current_run_id = f"exp3_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Get or create event loop
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        # Create run in manager
        future = asyncio.run_coroutine_threadsafe(
            live_run_manager.create_run(self.current_run_id),
            self._loop
        )
        future.result()  # Wait for completion
        
        return self.current_run_id
    
    def emit_step(self, step_data: dict) -> None:
        """Emit a step to all connected clients."""
        if not self.current_run_id or not self._loop:
            return
        
        # Send step asynchronously
        future = asyncio.run_coroutine_threadsafe(
            live_run_manager.add_step(self.current_run_id, step_data),
            self._loop
        )
        # Don't wait - let it run in background
    
    def end_run(self) -> None:
        """End the current simulation run."""
        if not self.current_run_id or not self._loop:
            return
        
        future = asyncio.run_coroutine_threadsafe(
            live_run_manager.end_run(self.current_run_id),
            self._loop
        )
        future.result()  # Wait for completion
        
        self.current_run_id = None


# Global instance for use in run.py
simulation_bridge = SimulationBridge()