"""Service for running simulations in-process within FastAPI."""
import asyncio
from typing import Dict, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import multiprocessing, signal
from multiprocessing.managers import SyncManager
import queue
from datetime import datetime
import uuid
import logging
import sys
import os

# Add backend to path so we can import simulation modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.application.services.event_aggregator import event_aggregator
from app.application.services.live_run_manager import live_run_manager
from app.api.v1.schemas.events import SimulationEvent
from app.core.parameter_service import ParameterService, SimulationConfig

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── pool / manager – built lazily ─────────────────────────────────────────
# Workers must **ignore SIGINT** so the parent alone handles Ctrl-C.
def _worker_ignore_sigint() -> None:          # moved to module scope → picklable
    signal.signal(signal.SIGINT, signal.SIG_IGN)

_EXECUTOR: Optional[ProcessPoolExecutor] = None
_MANAGER:  Optional[SyncManager] = None

def init_mp() -> None:
    """Create Manager + ProcessPool once (parent only)."""
    global _EXECUTOR, _MANAGER
    if _EXECUTOR is not None:          # already initialised
        return

    ctx = multiprocessing.get_context("spawn")
    _EXECUTOR = ProcessPoolExecutor(
        max_workers=2,
        mp_context=ctx,
        initializer=_worker_ignore_sigint,   # ← now safe to pickle
    )
    _MANAGER = ctx.Manager()


class SimulationService:
    """Service for managing in-process simulations."""
    
    def __init__(self):
        self.event_aggregator = event_aggregator
        self.live_run_manager = live_run_manager
        self.active_simulations: Dict[str, Dict[str, Any]] = {}
        self.result_queues: Dict[str, Any] = {}
        # keep an explicit handle to every consumer Task we spawn
        self._consumers: set[asyncio.Task] = set()
    
    async def start_simulation(self, params: Optional[Dict[str, Any]] = None) -> str:
        """Start a new simulation run in separate process to avoid blocking event loop."""
        # Ensure multiprocessing is initialized
        init_mp()
        
        # Generate run_id matching existing format
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"exp3_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting simulation {run_id} in separate process")
        
        # Register with LiveRunManager
        await self.live_run_manager.create_run(run_id)
        
        # Create multiprocessing manager queue for communication (pickle-safe)
        # maxsize=0 ⇒ un-bounded
        result_queue = _MANAGER.Queue(maxsize=0)
        self.result_queues[run_id] = result_queue
        
        # Start simulation in separate process
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            _EXECUTOR,
            _run_simulation_process,  # Pure sync function
            run_id,
            params,
            result_queue
        )
        
        # Start task to consume results from queue
        consumer_task = asyncio.create_task(
            self._consume_simulation_results(run_id, result_queue)
        )
        # remember it
        self._consumers.add(consumer_task)
        
        # Store both future and consumer task
        self.active_simulations[run_id] = {
            'future': future,
            'consumer': consumer_task
        }
        
        return run_id
    
    async def _consume_simulation_results(self, run_id: str, result_queue):
        """Consume results from simulation process and forward to live run manager."""
        try:
            while True:
                try:
                    # Non-blocking get with short timeout
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, result_queue.get, True, 0.1
                    )
                    
                    if result is None:  # Sentinel value for completion
                        break
                    
                    if result['type'] == 'frame':
                        await self.live_run_manager.add_step(run_id, result['data'])
                    elif result['type'] == 'event':
                        event_data = result['data']
                        event = SimulationEvent(**event_data)
                        await self.event_aggregator.add_event(run_id, event)
                    elif result['type'] == 'log':
                        logger.info(f"[Sim {run_id}] {result['message']}")
                        
                except queue.Empty:
                    # Check if simulation is still running
                    if run_id in self.active_simulations:
                        sim_info = self.active_simulations[run_id]
                        if sim_info['future'].done():
                            # Simulation finished, check for any remaining items
                            try:
                                while True:
                                    result = result_queue.get_nowait()
                                    if result is None:
                                        break
                                    # Process remaining items
                                    if result['type'] == 'frame':
                                        await self.live_run_manager.add_step(run_id, result['data'])
                                    elif result['type'] == 'event':
                                        event_data = result['data']
                                        event = SimulationEvent(**event_data)
                                        await self.event_aggregator.add_event(run_id, event)
                            except queue.Empty:
                                break
                            break
                    await asyncio.sleep(0.01)  # Small delay before next check
                    
        except Exception as e:
            logger.error(f"Error consuming simulation results for {run_id}: {e}")
        finally:
            await self.live_run_manager.end_run(run_id)
            # Clean up
            self.result_queues.pop(run_id, None)
            self.active_simulations.pop(run_id, None)
            # remove finished consumer from the registry
            self._consumers.discard(asyncio.current_task())
            # NOTE: Manager is shut down once, in app-shutdown
    
    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Get the status of a simulation run."""
        if run_id not in self.active_simulations:
            return {"status": "not_found"}
        
        sim_info = self.active_simulations[run_id]
        future = sim_info['future']
        
        if future.done():
            if future.exception():
                return {"status": "error", "error": str(future.exception())}
            else:
                return {"status": "completed"}
        else:
            return {"status": "running"}
    
    async def stop_simulation(self, run_id: str) -> bool:
        """Stop a running simulation."""
        if run_id not in self.active_simulations:
            return False
        
        sim_info = self.active_simulations[run_id]
        sim_info['future'].cancel()
        sim_info['consumer'].cancel()
        
        try:
            await sim_info['consumer']
        except asyncio.CancelledError:
            pass
        
        return True

    async def shutdown(self) -> None:
        """
        Cancel **all** in-flight simulations/consumer tasks so that
        Uvicorn's shutdown phase doesn't wait forever.
        """
        # 1) cancel futures still executing in the pool
        for info in self.active_simulations.values():
            info["future"].cancel()
        
        # 2) cancel **every** consumer coroutine we recorded
        for task in list(self._consumers):
            task.cancel()

        # 3) wait (briefly) until they acknowledge cancellation
        await asyncio.gather(*self._consumers, return_exceptions=True)

        self._consumers.clear()
        self.active_simulations.clear()
        self.result_queues.clear()


# Pure synchronous function to run in separate process
def _run_simulation_process(run_id: str, params: Optional[Dict[str, Any]], result_queue):
    """Run simulation in separate process - pure sync, no async/await."""
    try:
        # Import here to avoid issues with multiprocessing
        import sys
        import os
        import numpy as np
        from datetime import datetime
        
        # Add backend to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        
        from timer import Timer
        from environment import Environment
        from environment_behaviour import EnvironmentBehaviour
        from agent_behaviour import AgentsBehaviour
        from maliciousagent_behaviour import MaliciousAgentsBehaviour
        from serviceprovider_behaviour import ServiceProvidersBehaviour
        from app.core.parameter_service import ParameterService, SimulationConfig
        
        result_queue.put({'type': 'log', 'message': f'Starting simulation with PID {os.getpid()}'})
        
        # Parse parameters
        config = ParameterService.parse_parameters(params)
        
        # Validate configuration
        errors = ParameterService.validate_config(config)
        if errors:
            error_msg = "Invalid configuration: " + "; ".join(errors)
            result_queue.put({'type': 'log', 'message': error_msg})
            raise ValueError(error_msg)
        
        # Extract scenario flags for compatibility
        cyber_attack_enabled = "cyberAttack" in config.scenarios
        misinformation_enabled = "misinformation" in config.scenarios
        coordinated_attack_enabled = "coordinatedAttack" in config.scenarios
        
        timer = Timer()
        timer.start()
        
        # Phase 1: Run core experiment
        result_queue.put({'type': 'log', 'message': 'Running experiment phase...'})
        
        # Create environment and network
        env = Environment(config.num_agents, config.num_malicious_agents, config.num_providers, config.num_timesteps)
        network, neighbours = env.form_network(config.num_neighbours, config.rewiring_probability)
        cyber_matrix = np.full((config.num_timesteps, config.num_agents), -1, dtype=int)
        agents, regagents, malagents = env.create_agents()
        providers = env.create_providers()
        env.create_attributes()
        
        # Get provider reliability arrays
        endpoint_failure_rates = []
        endpoint_recovery_rates = []
        core_failure_rates = []
        core_recovery_rates = []
        
        for i in range(config.num_providers):
            provider_params = ParameterService.get_provider_params(config, i)
            endpoint_failure_rates.append(provider_params["endpoint_failure_rate"])
            endpoint_recovery_rates.append(provider_params["endpoint_recovery_rate"])
            core_failure_rates.append(provider_params["core_failure_rate"])
            core_recovery_rates.append(provider_params["core_recovery_rate"])
        
        # Create behavior objects
        env_behaviour = EnvironmentBehaviour(agents, regagents, malagents, providers, neighbours, network, config.num_timesteps)
        reg_behaviour = AgentsBehaviour(network, providers)
        serv_behaviour = ServiceProvidersBehaviour(providers, agents, core_failure_rates, core_recovery_rates, 
                                                   endpoint_failure_rates, endpoint_recovery_rates, config.num_timesteps)
        mal_behaviour = MaliciousAgentsBehaviour(network, malagents, providers, config.initial_q_value, config.num_timesteps)
        
        # Data collection arrays
        countUsers = np.zeros((config.num_timesteps+1, config.num_providers), dtype=int)
        countRewards = np.zeros((config.num_timesteps+1, config.num_providers), dtype=int)
        opinionValues = np.zeros((config.num_timesteps+1, len(agents), 2*len(providers)), dtype=float)
        actions, rewards, regrets, opinionvalues = [], [], [], []
        attack_decisions, targets, attack_methods = [], [], []
        attack_rewards, detection_rewards = [], []
        
        timestep = config.timestep
        
        # Main experiment loop
        for step in range(config.num_timesteps):
            # Environment
            env_behaviour.show_state(timestep)
            attack_reward, detection_reward = env_behaviour.calculate_malreward(config.misinfo_detection_rate, 
                                                                               config.cyber_detection_rate, timestep)
            
            # Malicious agents
            mal_behaviour.receive_malreward(attack_reward, detection_reward, timestep)
            mal_behaviour.update_malQ(config.learning_rate, config.window_size, timestep)
            attack_decision = mal_behaviour.select_action(cyber_attack_enabled, misinformation_enabled, 
                                                         coordinated_attack_enabled, config.window_size, timestep)
            target = mal_behaviour.select_target(attack_decision, cyber_attack_enabled, misinformation_enabled, 
                                               config.window_size, timestep)
            attack_method = mal_behaviour.select_attack_method(attack_decision, target, config.exploration_rate,
                                                              cyber_attack_enabled, misinformation_enabled, 
                                                              coordinated_attack_enabled, config.window_size, timestep)
            
            # Service providers
            for provider in providers:
                center = serv_behaviour.central_state(provider, target, attack_method, attack_decision, timestep)
                endpoint = serv_behaviour.endpoint_level(provider, center, timestep)
            
            # All agents   
            for agent in agents:
                action = reg_behaviour.select_action(agent, config.exploration_rate, config.experience_weight, timestep)
                cyber_matrix[step, agent] = action
                reward = reg_behaviour.receive_reward(agent, action, endpoint, timestep)
                reg_behaviour.update_Q(agent, action, reward, config.learning_rate)
                opinion = reg_behaviour.express_opinion(agent, config.exploration_rate, timestep)
                neighbour = reg_behaviour.find_neighbour(agent, timestep)
                feedback = reg_behaviour.ask_info(neighbour, malagents, mal_behaviour, opinion, config.satisfaction_threshold, attack_decision, attack_method)
                opinion_values = reg_behaviour.update_opinion_value(agent, opinion, feedback, config.learning_rate)
                
                countUsers[timestep, action] += 1
                countRewards[timestep, action] += reward
                opinionValues[timestep, agent] = opinion_values
            
            # Environment collects data
            env_behaviour.collect_data(countUsers, center, attack_decision, target, attack_method, timestep)
            avg_opinions = env_behaviour.show_opinion_dynamics(opinionValues, timestep)        
            regret = env_behaviour.calculate_regret(endpoint, timestep)
            
            # Collect data
            actions.append(countUsers[timestep])
            rewards.append(countRewards[timestep])
            opinionvalues.append(avg_opinions)
            regrets.append(regret)
            attack_decisions.append(attack_decision)
            targets.append(target)
            attack_methods.append(attack_method)
            attack_rewards.append(sum(attack_reward))
            detection_rewards.append(sum(detection_reward))
            
            timestep += 1
        
        # Convert to numpy arrays
        actions = np.array(actions)
        rewards = np.array(rewards)
        opinionvalues = np.array(opinionvalues)
        attack_decisions = np.array(attack_decisions)
        targets = np.array(targets)
        attack_methods = np.array(attack_methods)
        
        # Phase 2: Process and stream results
        result_queue.put({'type': 'log', 'message': 'Starting streaming phase...'})
        
        # Setup attack data
        mal_behaviour.attack_decisions = attack_decisions
        mal_behaviour.targets = targets
        mal_behaviour.attack_methods = attack_methods
        
        # Process attack events
        current_attack_interval = None
        typ_map = {0: "CYBER", 1: "MISINFO", 2: "COMBO"}

        # ─── provider-down tracking state ─────────────────────────
        prev_zd = {p: 1 for p in range(config.num_providers)}
        down_intervals = {}
        
        for t in range(config.num_timesteps):
            # Process attack intervals
            if t < len(attack_decisions) and attack_decisions[t] == 1:
                attack_method = attack_methods[t]
                current_attack_type = typ_map[int(attack_method)]
                
                if current_attack_interval is None:
                    current_attack_interval = {"type": current_attack_type, "start": t + 1, "end": None}
                elif current_attack_interval["type"] != current_attack_type:
                    # End previous interval
                    current_attack_interval["end"] = t
                    duration = current_attack_interval["end"] - current_attack_interval["start"] + 1
                    
                    event_data = {
                        't': current_attack_interval["start"],
                        'type': current_attack_interval["type"],
                        'duration': duration
                    }
                    result_queue.put({'type': 'event', 'data': event_data})
                    
                    # Start new interval
                    current_attack_interval = {"type": current_attack_type, "start": t + 1, "end": None}
            else:
                if current_attack_interval is not None:
                    # End current interval
                    current_attack_interval["end"] = t
                    duration = current_attack_interval["end"] - current_attack_interval["start"] + 1
                    
                    event_data = {
                        't': current_attack_interval["start"],
                        'type': current_attack_interval["type"],
                        'duration': duration
                    }
                    result_queue.put({'type': 'event', 'data': event_data})
                    current_attack_interval = None
            
            # Process provider events
            attack_info = mal_behaviour.get_active_attack_info(t)
            
            # Build frame data
            if t % 1 == 0:  # Every timestep
                # Build service edges
                current_service_edges = set()
                for agent in range(config.num_agents):
                    provider = cyber_matrix[t, agent]
                    if provider != -1:
                        edge = (str(agent), f"P-{provider}")
                        current_service_edges.add(edge)
                
                # Get attack edges
                current_attack_edges = set()
                attack_edges_data = mal_behaviour.generate_attack_edges(t)
                for edge_data in attack_edges_data:
                    edge_tuple = (edge_data["source"], edge_data["target"])
                    current_attack_edges.add((edge_tuple, edge_data["phase"]))
                
                # Build complete edge list
                all_edges = []
                
                # Info edges
                for src_node in network.nodes():
                    for tgt_node in network.neighbors(src_node):
                        if int(src_node) < int(tgt_node):
                            all_edges.append({
                                "id": f"{src_node}-{tgt_node}",
                                "source": str(src_node),
                                "target": str(tgt_node),
                                "layer": "info",
                                "phase": "NONE",
                                "dir": "↔",
                                "type": "info"
                            })
                
                # Service edges
                for s, tgt in current_service_edges:
                    all_edges.append({
                        "id": f"{s}-{tgt}",
                        "source": s,
                        "target": tgt,
                        "layer": "service",
                        "phase": "NONE",
                        "dir": "→",
                        "type": "service"
                    })
                
                # Attack edges
                for (s, tgt), phase in current_attack_edges:
                    all_edges.append({
                        "id": f"attack-{s}-{tgt}",
                        "source": s,
                        "target": tgt,
                        "layer": "attack",
                        "phase": phase,
                        "dir": "→",
                        "type": "attack"
                    })
                
                # Build nodes
                current_nodes = {}
                
                # Agent nodes
                for agent in range(config.num_agents):
                    provider = cyber_matrix[t, agent]
                    if provider != -1:
                        n_here = int(actions[t, provider])
                        r_per_user = rewards[t, provider] / n_here if n_here > 0 else 0
                        
                        ld = 0 if (attack_info.get("active") and 
                                  attack_info.get("target") == provider and 
                                  attack_info.get("method") in [0, 2]) else r_per_user
                        
                        sat = 1 if ld >= 0.8 else 0
                        det_cyber = 1 if (attack_info.get("active") and attack_info.get("target") == provider and 
                                        attack_info.get("method") in [0, 2] and ld == 0 and 
                                        np.random.random() <= config.cyber_detection_rate) else 0
                        det_misinfo = 0
                        
                        current_nodes[str(agent)] = {
                            "id": str(agent),
                            "prov": provider,
                            "reward": r_per_user,
                            "mal": int(agent in malagents),
                            "opin": opinionvalues[t, provider + config.num_providers] if (provider + config.num_providers) < opinionvalues.shape[1] else 0,
                            "ld": ld,
                            "sat": sat,
                            "detCyber": det_cyber,
                            "detMisinfo": det_misinfo
                        }
                
                # Provider nodes
                serv_behaviour.calculate_sentiment(network, t)
                provider_stats = serv_behaviour.get_provider_stats(t)

                # -- compare zd and raise events
                zd_now = {}
                for prov in range(config.num_providers):
                    zd = 0 if (attack_info.get("active")
                               and attack_info.get("target") == prov
                               and attack_info.get("method") in (0, 2)) else 1
                    zd_now[prov] = zd

                    if prev_zd[prov] == 1 and zd == 0:           # went DOWN
                        down_intervals[prov] = {"start": t + 1, "end": None}

                    if prev_zd[prov] == 0 and zd == 1:           # came UP
                        interval = down_intervals.get(prov)
                        if interval and interval["end"] is None:
                            interval["end"] = t
                            result_queue.put({
                                "type": "event",
                                "data": {
                                    "t": interval["start"],
                                    "type": "PROVIDER_DOWN",
                                    "provider": prov,
                                    "duration":
                                        interval["end"] - interval["start"] + 1
                                }
                            })
                    prev_zd[prov] = zd
                
                for prov in range(config.num_providers):
                    provider_id = f"P-{prov}"
                    zd = zd_now[prov]  # Re-use the already computed value
                    
                    current_nodes[provider_id] = {
                        "id": provider_id,
                        "prov": -1,
                        "reward": 1.0,
                        "mal": 0,
                        "opin": 0,
                        "zd": zd,
                        "sentiment": provider_stats[provider_id]["sentiment"]
                    }
                
                # Generate pulses
                pulses = env_behaviour.generate_info_pulses(reg_behaviour, t)
                
                # Create frame data
                frame_data = {
                    "t": t + 1,
                    "nodes": list(current_nodes.values()),
                    "edges": all_edges,
                    "provStats": provider_stats,
                    "pulses": pulses
                }
                
                result_queue.put({'type': 'frame', 'data': frame_data})
                
                if t <= 5 or t % 100 == 0:
                    result_queue.put({'type': 'log', 'message': f'Completed timestep {t}'})
        
        # Handle final attack interval
        if current_attack_interval is not None:
            current_attack_interval["end"] = config.num_timesteps
            duration = current_attack_interval["end"] - current_attack_interval["start"] + 1
            event_data = {
                't': current_attack_interval["start"],
                'type': current_attack_interval["type"],
                'duration': duration
            }
            result_queue.put({'type': 'event', 'data': event_data})
        
        # Close any provider-down intervals that never came back up
        for prov, interval in down_intervals.items():
            if interval["end"] is None:
                interval["end"] = config.num_timesteps
                result_queue.put({
                    "type": "event",
                    "data": {
                        "t": interval["start"],
                        "type": "PROVIDER_DOWN",
                        "provider": prov,
                        "duration": interval["end"] - interval["start"] + 1
                    }
                })

        timer.stop()
        result_queue.put({'type': 'log', 'message': f'Simulation completed in {timer.elapsed} seconds'})
        
        # Send sentinel to indicate completion
        result_queue.put(None)
        
    except Exception as e:
        result_queue.put({'type': 'log', 'message': f'Simulation failed: {str(e)}'})
        result_queue.put(None)
        raise


# Global instance
simulation_service = SimulationService()