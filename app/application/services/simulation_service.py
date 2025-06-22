"""Service for running simulations in-process within FastAPI."""
import asyncio
from typing import Dict, Any, Optional, Union
import concurrent.futures
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

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimulationService:
    """Service for managing in-process simulations."""
    
    def __init__(self):
        self.event_aggregator = event_aggregator
        self.live_run_manager = live_run_manager
        self.active_simulations: Dict[str, Union[asyncio.Task, concurrent.futures.Future]] = {}
    
    async def start_simulation(self, params: Optional[Dict[str, Any]] = None) -> str:
        """Start a new simulation run in background thread to avoid blocking event loop."""
        # Generate run_id matching existing format
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"exp3_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting simulation {run_id} in background thread")
        
        # Register with LiveRunManager
        await self.live_run_manager.create_run(run_id)
        
        # Run simulation in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        # Create a new event loop for the thread
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(self._run_simulation(run_id, params))
            finally:
                new_loop.close()
        
        # Execute in thread pool
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, run_in_thread)
        
        # Store the future
        self.active_simulations[run_id] = future
        
        return run_id
    
    def _run_experiment_phase(self, sv):
        """Run the core ABM simulation experiment (extracted from experiment.py)."""
        from environment import Environment
        from environment_behaviour import EnvironmentBehaviour
        from agent_behaviour import AgentsBehaviour
        from maliciousagent_behaviour import MaliciousAgentsBehaviour
        from serviceprovider_behaviour import ServiceProvidersBehaviour
        import numpy as np
        
        # Create environment and network
        env = Environment(sv.nAgents, sv.nMalAgents, sv.nProviders, sv.nTimesteps)
        network, neighbours = env.form_network(sv.kappa, sv.rho)
        cyber_matrix = np.full((sv.nTimesteps, sv.nAgents), -1, dtype=int)
        agents, regagents, malagents = env.create_agents()
        providers = env.create_providers()
        env.create_attributes()
        
        # Create behavior objects
        env_behaviour = EnvironmentBehaviour(agents, regagents, malagents, providers, neighbours, network, sv.nTimesteps)
        reg_behaviour = AgentsBehaviour(network, providers)
        serv_behaviour = ServiceProvidersBehaviour(providers, agents, sv.Upsilon, sv.upsilon_, sv.Lambda, sv.lambda_, sv.nTimesteps)
        mal_behaviour = MaliciousAgentsBehaviour(network, malagents, providers, sv.xi, sv.nTimesteps)
        
        # Data collection arrays
        countUsers = np.zeros((sv.nTimesteps+1, sv.nProviders), dtype=int)
        countRewards = np.zeros((sv.nTimesteps+1, sv.nProviders), dtype=int)
        opinionValues = np.zeros((sv.nTimesteps+1, len(agents), 2*len(providers)), dtype=float)
        actions, rewards, regrets, opinionvalues = [], [], [], []
        attack_decisions, targets, attack_methods = [], [], []
        attack_rewards, detection_rewards = [], []
        
        timestep = sv.timestep
        
        # Main experiment loop
        for step in range(sv.nTimesteps):
            # Environment
            env_behaviour.show_state(timestep)
            attack_reward, detection_reward = env_behaviour.calculate_malreward(sv.eta, sv.theta, timestep)
            
            # Malicious agents
            mal_behaviour.receive_malreward(attack_reward, detection_reward, timestep)
            mal_behaviour.update_malQ(sv.alpha, sv.W, timestep)
            attack_decision = mal_behaviour.select_action(sv.cyberattack, sv.misinformation, sv.coordinated_attack, sv.W, timestep)
            target = mal_behaviour.select_target(attack_decision, sv.cyberattack, sv.misinformation, sv.W, timestep)
            attack_method = mal_behaviour.select_attack_method(attack_decision, target, sv.epsilon, sv.cyberattack, sv.misinformation, sv.coordinated_attack, sv.W, timestep)
            
            # Service providers
            for provider in providers:
                center = serv_behaviour.central_state(provider, target, attack_method, attack_decision, timestep)
                endpoint = serv_behaviour.endpoint_level(provider, center, timestep)
            
            # All agents   
            for agent in agents:
                action = reg_behaviour.select_action(agent, sv.epsilon, sv.tau, timestep)
                cyber_matrix[step, agent] = action
                reward = reg_behaviour.receive_reward(agent, action, endpoint, timestep)
                reg_behaviour.update_Q(agent, action, reward, sv.alpha)
                opinion = reg_behaviour.express_opinion(agent, sv.epsilon, timestep)
                neighbour = reg_behaviour.find_neighbour(agent, timestep)
                feedback = reg_behaviour.ask_info(neighbour, malagents, mal_behaviour, opinion, sv.delta, attack_decision, attack_method)
                opinion_values = reg_behaviour.update_opinion_value(agent, opinion, feedback, sv.alpha)
                
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
        
        return {
            'env': env, 'network': network, 'cyber_matrix': cyber_matrix,
            'agents': agents, 'regagents': regagents, 'malagents': malagents,
            'providers': providers, 'behaviours': {
                'env': env_behaviour, 'reg': reg_behaviour, 
                'serv': serv_behaviour, 'mal': mal_behaviour
            },
            'results': {
                'actions': np.array(actions), 'rewards': np.array(rewards),
                'opinionvalues': np.array(opinionvalues), 'regrets': np.array(regrets),
                'attack_decisions': np.array(attack_decisions), 'targets': np.array(targets),
                'attack_methods': np.array(attack_methods), 'attack_rewards': np.array(attack_rewards),
                'detection_rewards': np.array(detection_rewards)
            }
        }
    
    async def _process_attack_events(self, run_id: str, t: int, attack_decisions, attack_methods, current_attack_interval):
        """Handle attack interval tracking and event emission."""
        if t >= len(attack_decisions):
            return current_attack_interval
            
        typ_map = {0: "CYBER", 1: "MISINFO", 2: "COMBO"}
        
        if attack_decisions[t] == 1:  # Attack is active
            attack_method = attack_methods[t]
            current_attack_type = typ_map[int(attack_method)]
            
            if current_attack_interval is None:
                current_attack_interval = {"type": current_attack_type, "start": t + 1, "end": None}
            elif current_attack_interval["type"] != current_attack_type:
                # End previous interval and emit event
                current_attack_interval["end"] = t
                duration = current_attack_interval["end"] - current_attack_interval["start"] + 1
                
                event = SimulationEvent(
                    t=current_attack_interval["start"],
                    type=current_attack_interval["type"],
                    duration=duration
                )
                await self.event_aggregator.add_event(run_id, event)
                
                # Start new interval
                current_attack_interval = {"type": current_attack_type, "start": t + 1, "end": None}
        else:  # No attack active
            if current_attack_interval is not None:
                # End current attack interval
                current_attack_interval["end"] = t
                duration = current_attack_interval["end"] - current_attack_interval["start"] + 1
                
                event = SimulationEvent(
                    t=current_attack_interval["start"],
                    type=current_attack_interval["type"],
                    duration=duration
                )
                await self.event_aggregator.add_event(run_id, event)
                current_attack_interval = None
        
        return current_attack_interval
    
    def _process_provider_events(self, run_id: str, t: int, attack_info, sv, prev_provider_states, provider_down_intervals):
        """Handle provider state tracking for down events."""
        events_to_emit = []
        
        for prov in range(sv.nProviders):
            # Get central state (zd) - 0 if under cyber attack, 1 otherwise  
            zd = 0 if (attack_info.get("active") and 
                      attack_info.get("target") == prov and 
                      attack_info.get("method") in [0, 2]) else 1
            
            prev_zd = prev_provider_states.get(prov, 1)
            
            if prev_zd == 1 and zd == 0:  # Provider goes down
                provider_down_intervals[prov] = {"start": t + 1, "end": None}
            elif prev_zd == 0 and zd == 1:  # Provider comes back up
                if prov in provider_down_intervals and provider_down_intervals[prov]["end"] is None:
                    interval = provider_down_intervals[prov]
                    interval["end"] = t
                    
                    event = SimulationEvent(
                        t=interval["start"], 
                        type="PROVIDER_DOWN",
                        provider=prov,
                        duration=interval["end"] - interval["start"] + 1
                    )
                    events_to_emit.append(event)
            
            prev_provider_states[prov] = zd
        
        return events_to_emit

    async def _run_simulation(self, run_id: str, params: Optional[Dict[str, Any]]):
        """Main simulation method with proper separation of concerns."""
        try:
            logger.info(f"Simulation {run_id} starting with PID {os.getpid()}")
            
            # Import simulation components
            from timer import Timer
            import setup_values as sv
            import numpy as np
            
            timer = Timer()
            timer.start()
            
            # Phase 1: Run core experiment simulation (ONCE - no duplication)
            logger.info("Running experiment phase...")
            experiment_data = self._run_experiment_phase(sv)
            
            # Extract results for streaming phase
            behaviours = experiment_data['behaviours']
            results = experiment_data['results']
            
            # Phase 2: Process results for live streaming
            logger.info("Starting streaming phase...")
            
            # Setup attack data for streaming behaviors
            behaviours['mal'].attack_decisions = results['attack_decisions']
            behaviours['mal'].targets = results['targets']
            behaviours['mal'].attack_methods = results['attack_methods']
            
            # Tracking state
            prev_provider_states = {}
            provider_down_intervals = {}
            current_attack_interval = None
            
            # Edge tracking (EXACT same as run.py lines 235-242)
            prev_service_edges = set()
            prev_attack_edges = set()
            service_adds = set()
            service_removes = set()
            attack_adds = set()
            attack_removes = set()
            
            # Node delta tracking (EXACT same as run.py line 243)
            prev_nodes = {}
            
            def edge_id(src, tgt): 
                return f"{src}-{tgt}"
            
            # Main streaming loop
            for t in range(sv.nTimesteps):
                # Process attack events
                current_attack_interval = await self._process_attack_events(
                    run_id, t, results['attack_decisions'], results['attack_methods'], current_attack_interval
                )
                
                # Get attack info for this timestep
                attack_info = behaviours['mal'].get_active_attack_info(t)
                
                # Process provider events
                provider_events = self._process_provider_events(
                    run_id, t, attack_info, sv, prev_provider_states, provider_down_intervals
                )
                
                # Emit provider events
                for event in provider_events:
                    await self.event_aggregator.add_event(run_id, event)
                
                if t % 1 == 0:  # Export every timestep
                    # Build current service edges
                    current_service_edges = set()
                    for agent in range(sv.nAgents):
                        provider = experiment_data['cyber_matrix'][t, agent]
                        if provider != -1:
                            edge = (str(agent), f"P-{provider}")
                            current_service_edges.add(edge)
                    
                    # Get attack edges
                    current_attack_edges = set()
                    attack_edges_data = behaviours['mal'].generate_attack_edges(t)
                    for edge_data in attack_edges_data:
                        edge_tuple = (edge_data["source"], edge_data["target"])
                        current_attack_edges.add((edge_tuple, edge_data["phase"]))
                    
                    # Build complete edge list for live streaming
                    all_edges = []
                    
                    # Info edges from social network
                    for src_node in experiment_data['network'].nodes():
                        for tgt_node in experiment_data['network'].neighbors(src_node):
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
                            "id": edge_id(s, tgt),
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
                            "id": f"attack-{edge_id(s, tgt)}",
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
                    for agent in range(sv.nAgents):
                        provider = experiment_data['cyber_matrix'][t, agent]
                        if provider != -1:
                            n_here = int(results['actions'][t, provider])
                            r_per_user = results['rewards'][t, provider] / n_here if n_here > 0 else 0
                            
                            ld = 0 if (attack_info.get("active") and 
                                      attack_info.get("target") == provider and 
                                      attack_info.get("method") in [0, 2]) else r_per_user
                            
                            sat = 1 if ld >= 0.8 else 0
                            det_cyber = 1 if (attack_info.get("active") and attack_info.get("target") == provider and 
                                            attack_info.get("method") in [0, 2] and ld == 0 and 
                                            np.random.random() <= sv.theta) else 0
                            det_misinfo = 0  # Simplified for now
                            
                            current_nodes[str(agent)] = {
                                "id": str(agent),
                                "prov": provider,
                                "reward": r_per_user,
                                "mal": int(agent in experiment_data['malagents']),
                                "opin": results['opinionvalues'][t, provider + sv.nProviders] if (provider + sv.nProviders) < results['opinionvalues'].shape[1] else 0,
                                "ld": ld,
                                "sat": sat,
                                "detCyber": det_cyber,
                                "detMisinfo": det_misinfo
                            }
                    
                    # Provider nodes
                    behaviours['serv'].calculate_sentiment(experiment_data['network'], t)
                    provider_stats = behaviours['serv'].get_provider_stats(t)
                    
                    for prov in range(sv.nProviders):
                        provider_id = f"P-{prov}"
                        zd = 0 if (attack_info.get("active") and 
                                  attack_info.get("target") == prov and 
                                  attack_info.get("method") in [0, 2]) else 1
                        
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
                    pulses = behaviours['env'].generate_info_pulses(behaviours['reg'], t)
                    
                    # Create frame data in EXACT same format as run.py (live streaming version)
                    frame_data = {
                        "t": t + 1,
                        "nodes": list(current_nodes.values()),  # Send ALL nodes for live streaming
                        "edges": all_edges,  # Send ALL edges for live streaming
                        "provStats": provider_stats,
                        "pulses": pulses
                    }
                    
                    await self.live_run_manager.add_step(run_id, frame_data)
                    
                    if t <= 5 or t % 100 == 0:
                        logger.info(f"Completed timestep {t}")
                
                # Yield control every timestep to prevent blocking FastAPI
                await asyncio.sleep(0)
            
            # Handle final intervals
            if current_attack_interval is not None:
                current_attack_interval["end"] = sv.nTimesteps
                duration = current_attack_interval["end"] - current_attack_interval["start"] + 1
                event = SimulationEvent(
                    t=current_attack_interval["start"],
                    type=current_attack_interval["type"],
                    duration=duration
                )
                await self.event_aggregator.add_event(run_id, event)
            
            for prov, interval in provider_down_intervals.items():
                if interval["end"] is None:
                    interval["end"] = sv.nTimesteps
                    event = SimulationEvent(
                        t=interval["start"], 
                        type="PROVIDER_DOWN",
                        provider=prov,
                        duration=interval["end"] - interval["start"] + 1
                    )
                    await self.event_aggregator.add_event(run_id, event)
            
            timer.stop()
            await self.live_run_manager.end_run(run_id)
            logger.info(f"Simulation {run_id} completed in {timer.elapsed} seconds")
            
        except Exception as e:
            logger.error(f"Simulation {run_id} failed: {e}", exc_info=True)
            try:
                await self.live_run_manager.end_run(run_id)
            except:
                pass
            raise
    
    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Get the status of a simulation run."""
        if run_id not in self.active_simulations:
            return {"status": "not_found"}
        
        task = self.active_simulations[run_id]
        if task.done():
            if task.exception():
                return {"status": "error", "error": str(task.exception())}
            else:
                return {"status": "completed"}
        else:
            return {"status": "running"}
    
    async def stop_simulation(self, run_id: str) -> bool:
        """Stop a running simulation."""
        if run_id not in self.active_simulations:
            return False
        
        task = self.active_simulations[run_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        return True


# Global instance
simulation_service = SimulationService()