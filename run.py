# run.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Andreas Sjöstedt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# ---------------------------------------------------------------------------
#  ⚙️  Hybrid‑Article main driver  (stream-friendly, O(1)-RAM rewrite) (full original logic + perf & head‑less fixes)
# ---------------------------------------------------------------------------

import os, gzip, json, warnings, logging
from datetime import datetime, timezone
import matplotlib, matplotlib.pyplot as plt
import numpy as np, pandas as pd, orjson

# Set up debug logging for simulation events
debug_logger = logging.getLogger('simulation_debug')
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler('simulation_debug.log', mode='w')
debug_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
debug_logger.addHandler(debug_handler)

from timer import Timer
from experiment import experiment
from environment import Environment
from environment_behaviour import EnvironmentBehaviour
from agent_behaviour import AgentsBehaviour
from serviceprovider_behaviour import ServiceProvidersBehaviour
from maliciousagent_behaviour import MaliciousAgentsBehaviour
from setup_values import (
    timestep, nTimesteps, nExperiments, nProviders, nAgents, nMalAgents, W,
    save_fig, output_dir, alpha, delta, epsilon, eta, theta, kappa, Lambda,
    lambda_ as lambda_arr, xi, rho, tau, Upsilon, upsilon_, experiment_id,
)

# Import simulation bridge for live streaming
import queue
import threading
try:
    import requests
    LIVE_STREAMING = True
    # Test if FastAPI server is running
    try:
        requests.get("http://localhost:8000/", timeout=1)
        print("FastAPI server detected - live streaming enabled")
        debug_logger.info("FastAPI server detected - live streaming enabled")
    except:
        LIVE_STREAMING = False
        print("FastAPI server not running - live streaming disabled")
        debug_logger.warning("FastAPI server not running - live streaming disabled")
except ImportError:
    LIVE_STREAMING = False
    print("requests not available - live streaming disabled")
    debug_logger.warning("requests not available - live streaming disabled")

# Import event aggregator (always import at module level for singleton consistency)
try:
    from app.application.services.event_aggregator import event_aggregator
    from app.api.v1.schemas.events import SimulationEvent
    
    if LIVE_STREAMING:
        EVENT_AGGREGATION = True
        debug_logger.info("Event aggregation enabled successfully")
        
        # Create thread-safe event queue and background processor
        event_queue = queue.Queue()
        
        def event_processor():
            """Background thread to process events without blocking simulation"""
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            debug_logger.info("Event processor thread started")
            
            while True:
                try:
                    run_id, event = event_queue.get(timeout=1)
                    if run_id is None:  # Shutdown signal
                        break
                    debug_logger.info(f"Event processor: Got event from queue: {event.type} at t={event.t}")
                    loop.run_until_complete(event_aggregator.add_event(run_id, event))
                    debug_logger.info(f"Event processor: Event sent to aggregator successfully")
                    event_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    debug_logger.error(f"Event processor error: {e}")
            
            loop.close()
        
        # Start background event processor
        event_thread = threading.Thread(target=event_processor, daemon=True)
        event_thread.start()
        
        def emit_event_async(run_id, event):
            """Non-blocking event emission"""
            try:
                debug_logger.info(f"Emitting event: {event.type} at t={event.t} for run {run_id}")
                event_queue.put((run_id, event), block=False)
                debug_logger.debug(f"Event queued successfully: {event}")
            except queue.Full:
                debug_logger.error("Event queue full - dropping event")
    else:
        EVENT_AGGREGATION = False
        debug_logger.info("Event aggregation disabled - live streaming not enabled")
        
        def emit_event_async(run_id, event):
            """Dummy function when event aggregation is disabled"""
            pass
        
except ImportError:
    EVENT_AGGREGATION = False
    print("Event aggregation not available")
    debug_logger.warning("Event aggregation not available - import failed")
    
    def emit_event_async(run_id, event):
        """Dummy function when event aggregation is not available"""
        pass

# Export every nth timestep
EXPORT_INTERVAL = 1

ORJSON_OPTS = (
    orjson.OPT_SERIALIZE_NUMPY
    | orjson.OPT_NAIVE_UTC
    | orjson.OPT_NON_STR_KEYS
)
def dumps(obj):          # keep it tiny
    return orjson.dumps(obj, option=ORJSON_OPTS)

def edge_id(src, tgt): 
    return f"{src}-{tgt}"

###############################################################################
# IO PATHS & CONSTANTS
###############################################################################
OUTPUT_DATA_DIR = os.path.join(os.getcwd(), "output_data")
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
SUMMARY_CSV  = os.path.join(OUTPUT_DATA_DIR, f"experiment{experiment_id}_summary.csv")
FRAMES_JSONL = os.path.join(
    OUTPUT_DATA_DIR,
    f"frames_exp{experiment_id}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.jsonl.gz",
)

# header carries meta + events so the UI can find everything
HEADER_JSON = FRAMES_JSONL.replace("frames_", "header_").replace(".jsonl.gz", ".json")

if save_fig:
    os.makedirs(output_dir, exist_ok=True)

# Start live streaming run if available
current_run_id = None
if LIVE_STREAMING:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    import uuid
    current_run_id = f"exp3_{timestamp}_{uuid.uuid4().hex[:8]}"
    print(f"Started live run: {current_run_id}")
    
    # Register run with FastAPI server
    try:
        response = requests.post(f"http://localhost:8000/api/v1/runs/{current_run_id}/register", timeout=5)
        print(f"Run registration response: {response.status_code}")
    except Exception as e:
        print(f"Failed to register run with server: {e}")
        LIVE_STREAMING = False

# Head‑less backend detection – "Agg", "PDF", etc. never open a window ------
HEADLESS_BACKENDS = {"agg", "pdf", "svg"}
SHOW_FIGS = (not save_fig) and matplotlib.get_backend().lower() not in HEADLESS_BACKENDS

def _maybe_show():
    """Show figure only if interactive; always close to free memory."""
    if SHOW_FIGS:
        plt.show()
    plt.close()
# ---------------------------------------------------------------------------

# All dataframe columns (kept verbatim from original) -----------------------
COLUMNS = [
    "regret", "provider", "avguse", "avgreward", "posopinion", "negopinion",
    "avgusebefore", "avguseduring", "avguseafter", "avgrewbefore", "avgrewduring",
    "avgrewafter", "posopinionbefore", "posopinionduring", "posopinionafter",
    "negopinionbefore", "negopinionduring", "negopinionafter", "target_id",
    "attack_beginning", "misinfo_beginning", "misinfo_length", "cyber_beginning",
    "cyber_length", "combined_beginning", "combined_length", "impactusage",
    "impactposop", "impactnegop", "attackreward", "detectionreward", "malreward",
    "alpha", "delta", "epsilon", "eta", "theta", "kappa", "Lambda", "lambda", "xi",
    "rho", "tau", "Upsilon", "upsilon",
]
# ---------------------------------------------------------------------------

run = 1  # You can bump this to run the same parameter set multiple times

for _run_idx in range(run):
    print(f"Starting simulation run {_run_idx + 1}/{run}")
    timer = Timer(); timer.start()

    rows:   list[dict] = []       # provider-level summary rows
    events: list[dict] = []       # attack timeline
    jsonl  = gzip.open(FRAMES_JSONL, "wb")   # ← stream frames here

    # ── MAIN EXPERIMENT LOOP ───────────────────────────────────────────────
    for exp in range(nExperiments):
        print(f"[Experiment {exp + 1}/{nExperiments}]")

        actions, rewards, opinionvalues, regrets, attack_decisions, targets, \
        attack_methods, attack_rewards, detection_rewards, links, cyber_matrix = experiment(timestep, nTimesteps)

        # Average regret after warm‑up window W
        avgregret = np.mean(regrets[W + 1:]) / nAgents

        # ── STREAM -- one JSONL per timestep with 4-layer edge architecture ──
        prev_service_edges = set()
        prev_attack_edges = set()
        service_adds = set()
        service_removes = set()
        attack_adds = set()  
        attack_removes = set()
        
        # Node delta tracking
        prev_nodes = {}
        
        # Create behavior objects with enhanced methods
        # (This is needed to access the new detection and sentiment tracking)
        env = Environment(nAgents, nMalAgents, nProviders, nTimesteps)
        network, neighbours = env.form_network(kappa, rho)
        agents_list, regagents, malagents = env.create_agents()
        providers_list = env.create_providers()
        env.create_attributes()
        
        env_behaviour = EnvironmentBehaviour(agents_list, regagents, malagents, providers_list, neighbours, network, nTimesteps)
        reg_behaviour = AgentsBehaviour(network, providers_list)
        serv_behaviour = ServiceProvidersBehaviour(providers_list, agents_list, Upsilon, upsilon_, Lambda, lambda_arr, nTimesteps)
        mal_behaviour = MaliciousAgentsBehaviour(network, malagents, providers_list, xi, nTimesteps)
        
        # Copy attack data from experiment results
        mal_behaviour.attack_decisions = attack_decisions
        mal_behaviour.targets = targets  
        mal_behaviour.attack_methods = attack_methods
        
        # Track provider states for transition event detection
        prev_provider_states = {}  # provider_id -> zd state (0=down, 1=up)
        provider_down_intervals = {}  # provider_id -> {"start": t, "end": None}
        
        # Track attack intervals for range-based events
        current_attack_interval = None  # {"type": "CYBER", "start": t, "end": None}
        
        for t in range(nTimesteps):
            if t == 0:
                print(f"Starting timestep loop, nTimesteps={nTimesteps}")
            # Build current service edges (agent -> provider connections)
            current_service_edges = set()
            for agent in range(nAgents):
                provider = cyber_matrix[t, agent]
                if provider != -1:  # agent is connected to a provider
                    edge = (str(agent), f"P-{provider}")
                    current_service_edges.add(edge)
            
            # Get attack edges from malicious behavior
            current_attack_edges = set()
            attack_edges_data = mal_behaviour.generate_attack_edges(t)
            
            for edge_data in attack_edges_data:
                edge_tuple = (edge_data["source"], edge_data["target"])
                current_attack_edges.add((edge_tuple, edge_data["phase"]))
            
            # Track service edge changes
            newly_added_service = current_service_edges - prev_service_edges
            newly_removed_service = prev_service_edges - current_service_edges
            
            # Track attack edge changes  
            newly_added_attack = current_attack_edges - prev_attack_edges
            newly_removed_attack = prev_attack_edges - current_attack_edges
            
            # DEBUG: Print attack edge generation and deltas
            if t >= 2900 and t <= 2920:
                print(f"DEBUG Frame {t}: Generated {len(attack_edges_data)} attack edges")
                if attack_edges_data:
                    print(f"  Sample edge: {attack_edges_data[0]}")
                    print(f"  Malicious agents: {sorted(malagents)}")
                    print(f"  Attack sources: {sorted(set(edge['source'] for edge in attack_edges_data))}")
                print(f"  Attack adds: {len(newly_added_attack)}, removes: {len(newly_removed_attack)}")
                if newly_added_attack:
                    print(f"  Add sample: {list(newly_added_attack)[0]}")
                if newly_removed_attack:
                    print(f"  Remove sample: {list(newly_removed_attack)[0]}")
            
            # Accumulate service edge changes
            for edge in newly_added_service:
                if edge in service_removes:
                    service_removes.discard(edge)
                else:
                    service_adds.add(edge)
            
            for edge in newly_removed_service:
                if edge in service_adds:
                    service_adds.discard(edge)
                else:
                    service_removes.add(edge)
            
            # Accumulate attack edge changes
            for edge in newly_added_attack:
                if edge in attack_removes:
                    attack_removes.discard(edge)
                else:
                    attack_adds.add(edge)
            
            for edge in newly_removed_attack:
                if edge in attack_adds:
                    attack_adds.discard(edge)
                else:
                    attack_removes.add(edge)
            
            prev_service_edges = current_service_edges
            prev_attack_edges = current_attack_edges
            
            # Export on interval
            if t % EXPORT_INTERVAL == 0:
                if t <= 5:  # Debug first few timesteps
                    debug_logger.info(f"Processing timestep {t}, LIVE_STREAMING={LIVE_STREAMING}, EVENT_AGGREGATION={EVENT_AGGREGATION}, run_id={current_run_id}")
                # Build current node state
                current_nodes = {}
                nodes_for_frame = []
                node_deltas = {"add": [], "remove": [], "update": []}
                
                # Simulate detection states and satisfaction for agents
                attack_info = mal_behaviour.get_active_attack_info(t)
                
                # Track attack intervals for range-based events
                if EVENT_AGGREGATION and current_run_id and t < len(attack_decisions):
                    typ_map = {0: "CYBER", 1: "MISINFO", 2: "COMBO"}
                    
                    if t <= 10 or t % 100 == 0:  # Log early timesteps and every 100th
                        debug_logger.debug(f"t={t}: attack_decision={attack_decisions[t] if t < len(attack_decisions) else 'N/A'}, current_interval={current_attack_interval}")
                    
                    if attack_decisions[t] == 1:  # Attack is active
                        attack_method = attack_methods[t]
                        current_attack_type = typ_map[int(attack_method)]
                        
                        if current_attack_interval is None:
                            # Start new attack interval
                            current_attack_interval = {
                                "type": current_attack_type,
                                "start": t + 1,
                                "end": None
                            }
                        elif current_attack_interval["type"] != current_attack_type:
                            # Attack type changed - end previous interval and start new one
                            current_attack_interval["end"] = t  # End at previous timestep
                            duration = current_attack_interval["end"] - current_attack_interval["start"] + 1
                            
                            # Emit completed interval
                            event = SimulationEvent(
                                t=current_attack_interval["start"],
                                type=current_attack_interval["type"],
                                duration=duration
                            )
                            if EVENT_AGGREGATION:
                                emit_event_async(current_run_id, event)
                            
                            # Start new interval
                            current_attack_interval = {
                                "type": current_attack_type,
                                "start": t + 1,
                                "end": None
                            }
                    
                    else:  # No attack active
                        if current_attack_interval is not None:
                            # End current attack interval
                            current_attack_interval["end"] = t  # End at previous timestep
                            duration = current_attack_interval["end"] - current_attack_interval["start"] + 1
                            
                            # Emit completed interval
                            event = SimulationEvent(
                                t=current_attack_interval["start"],
                                type=current_attack_interval["type"],
                                duration=duration
                            )
                            try:
                                import asyncio
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(event_aggregator.add_event(current_run_id, event))
                                loop.close()
                            except Exception as e:
                                print(f"Failed to emit {current_attack_interval['type']} event: {e}")
                            
                            current_attack_interval = None
                
                # Build current agent nodes
                for agent in range(nAgents):
                    provider = cyber_matrix[t, agent]
                    if provider != -1:  # agent is connected to a provider
                        # Calculate reward for this agent's provider
                        n_here = int(actions[t, provider])
                        r_per_user = rewards[t, provider] / n_here if n_here > 0 else 0
                        
                        # Calculate endpoint availability (ld) - 0 if under attack, reward otherwise
                        ld = 0 if (attack_info["active"] and 
                                  attack_info["target"] == provider and 
                                  attack_info["method"] in [0, 2]) else r_per_user
                        
                        # Calculate satisfaction (sat) - 1 if ld >= 0.8
                        sat = 1 if ld >= 0.8 else 0
                        
                        # Simulate detection states based on probabilities
                        det_cyber = 0
                        det_misinfo = 0
                        
                        if attack_info["active"] and attack_info["target"] == provider:
                            if attack_info["method"] in [0, 2] and ld == 0:  # Cyber attack affecting this agent
                                det_cyber = 1 if np.random.random() <= theta else 0
                            if attack_info["method"] in [1, 2]:  # Misinformation attack
                                # Check if agent communicated with malicious agent
                                communicated_with_malicious = (
                                    t > 0 and 
                                    "zeta" in network.nodes[agent] and 
                                    t < len(network.nodes[agent]["zeta"]) and
                                    network.nodes[agent]["zeta"][t] in malagents
                                )
                                if communicated_with_malicious:
                                    det_misinfo = 1 if np.random.random() <= eta else 0
                        
                        node_data = {
                            "id": str(agent),
                            "prov": provider,
                            "reward": r_per_user,
                            "mal": int(agent in malagents),
                            "opin": opinionvalues[t, provider + nProviders] if (provider + nProviders) < opinionvalues.shape[1] else 0,
                            # NEW FIELDS
                            "ld": ld,
                            "sat": sat,
                            "detCyber": det_cyber,
                            "detMisinfo": det_misinfo
                        }
                        current_nodes[str(agent)] = node_data
                
                # Calculate provider sentiment and add provider nodes
                serv_behaviour.calculate_sentiment(network, t)
                provider_stats = serv_behaviour.get_provider_stats(t)
                
                for prov in range(nProviders):
                    provider_id = f"P-{prov}"
                    
                    # Get central state (zd) - 0 if under cyber attack, 1 otherwise  
                    zd = 0 if (attack_info["active"] and 
                              attack_info["target"] == prov and 
                              attack_info["method"] in [0, 2]) else 1
                    
                    # Track provider state transitions for event emission
                    if EVENT_AGGREGATION and current_run_id:
                        prev_zd = prev_provider_states.get(prov, 1)  # Default to up
                        
                        if prev_zd == 1 and zd == 0:  # Provider goes down
                            provider_down_intervals[prov] = {"start": t + 1, "end": None}
                            
                        elif prev_zd == 0 and zd == 1:  # Provider comes back up
                            if prov in provider_down_intervals and provider_down_intervals[prov]["end"] is None:
                                # Complete the interval and emit event
                                interval = provider_down_intervals[prov]
                                interval["end"] = t  # End at previous timestep
                                
                                event = SimulationEvent(
                                    t=interval["start"], 
                                    type="PROVIDER_DOWN",
                                    provider=prov,
                                    duration=interval["end"] - interval["start"] + 1
                                )
                                if EVENT_AGGREGATION:
                                    emit_event_async(current_run_id, event)
                        
                        prev_provider_states[prov] = zd
                    
                    node_data = {
                        "id": provider_id,
                        "prov": -1,    # Special marker for provider nodes
                        "reward": 1.0,  # Frontend expects providers to have reward 1.0
                        "mal": 0,
                        "opin": 0,
                        # NEW FIELDS
                        "zd": zd,
                        "sentiment": provider_stats[provider_id]["sentiment"]
                    }
                    current_nodes[provider_id] = node_data
                
                # First frame: send all nodes, no deltas
                if not prev_nodes:
                    nodes_for_frame = list(current_nodes.values())
                else:
                    # Compare with previous state and build deltas
                    for node_id, node_data in current_nodes.items():
                        if node_id not in prev_nodes:
                            # New node
                            node_deltas["add"].append(node_data)
                        elif prev_nodes[node_id] != node_data:
                            # Changed node
                            node_deltas["update"].append(node_data)
                    
                    # Check for removed nodes
                    for node_id in prev_nodes:
                        if node_id not in current_nodes:
                            node_deltas["remove"].append({"id": node_id})
                    
                    # Only include changed nodes in frame
                    nodes_for_frame = node_deltas["add"] + node_deltas["update"]
                
                # Update previous state
                prev_nodes = current_nodes.copy()
                
                # Convert accumulated changes to new 4-layer edge format
                edge_add = []
                
                # Add service edges
                for s, tgt in service_adds:
                    edge_add.append({
                        "id": edge_id(s, tgt),
                        "source": s,
                        "target": tgt,
                        "layer": "service",
                        "phase": "NONE",
                        "dir": "→",
                        "type": "service"  # Backward compatibility
                    })
                
                # Add attack edges
                for (s, tgt), phase in attack_adds:
                    edge_add.append({
                        "id": f"attack-{edge_id(s, tgt)}",
                        "source": s,
                        "target": tgt,
                        "layer": "attack",
                        "phase": phase,
                        "dir": "→",
                        "type": "attack"  # Backward compatibility
                    })
                
                edge_remove = []
                
                # Remove service edges
                for s, tgt in service_removes:
                    edge_remove.append({
                        "id": edge_id(s, tgt),
                        "source": s,
                        "target": tgt,
                        "layer": "service"
                    })
                
                # Remove attack edges
                for (s, tgt), phase in attack_removes:
                    edge_remove.append({
                        "id": f"attack-{edge_id(s, tgt)}",
                        "source": s,
                        "target": tgt,
                        "layer": "attack",
                        "phase": phase
                    })
                
                # Generate info pulses for this timestep
                pulses = env_behaviour.generate_info_pulses(reg_behaviour, t)
                
                # Write enhanced JSONL line per frame
                frame_data = {
                    "t": t + 1,
                    "nodes": nodes_for_frame,
                    "edgeDeltas": {"add": edge_add, "remove": edge_remove},
                    "provStats": provider_stats,  # NEW: Provider statistics
                    "pulses": pulses              # NEW: Information flow pulses
                }
                
                # Add nodeDeltas for delta frames (not first frame)
                if prev_nodes and (node_deltas["add"] or node_deltas["remove"] or node_deltas["update"]):
                    frame_data["nodeDeltas"] = node_deltas
                
                jsonl.write(dumps(frame_data) + b"\n")
                
                # Store step for live stream if available (frontend will request chunks)
                if LIVE_STREAMING and current_run_id:
                    # For live streaming, send ALL edges (not deltas)
                    all_edges = []
                    
                    # Add static info edges from social network
                    for src_node in network.nodes():
                        for tgt_node in network.neighbors(src_node):
                            if int(src_node) < int(tgt_node):  # Avoid duplicates
                                all_edges.append({
                                    "id": f"{src_node}-{tgt_node}",
                                    "source": str(src_node),
                                    "target": str(tgt_node),
                                    "layer": "info",
                                    "phase": "NONE",
                                    "dir": "↔",
                                    "type": "info"
                                })
                    
                    # Add current service edges
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
                    
                    # Add current attack edges
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
                    
                    # For live streaming, send ALL nodes and ALL edges
                    live_frame_data = {
                        "t": t + 1,
                        "nodes": list(current_nodes.values()),  # Send ALL nodes
                        "edges": all_edges,  # Send ALL edges (not deltas)
                        "provStats": provider_stats,
                        "pulses": pulses
                    }
                    
                    # Convert numpy types to native Python types for JSON serialization
                    json_frame_data = json.loads(dumps(live_frame_data))
                    
                    try:
                        # Just add the step - frontend will request it via WebSocket
                        response = requests.post(
                            f"http://localhost:8000/api/v1/runs/{current_run_id}/step",
                            json=json_frame_data,
                            timeout=0.5
                        )
                        if t <= 5:  # Only log first few steps
                            print(f"Stored step {t}, status: {response.status_code}")
                    except Exception as e:
                        if t <= 5:  # Only log first few errors
                            print(f"Failed to store step {t}: {e}")
                        # Don't let streaming errors stop the simulation
                        pass
                
                # Reset accumulators after export
                service_adds.clear()
                service_removes.clear()
                attack_adds.clear()
                attack_removes.clear()
        
        # Handle any remaining intervals at end of simulation
        if EVENT_AGGREGATION and current_run_id:
            # Complete any ongoing attack interval
            if current_attack_interval is not None:
                current_attack_interval["end"] = nTimesteps
                duration = current_attack_interval["end"] - current_attack_interval["start"] + 1
                
                event = SimulationEvent(
                    t=current_attack_interval["start"],
                    type=current_attack_interval["type"],
                    duration=duration
                )
                if EVENT_AGGREGATION:
                    emit_event_async(current_run_id, event)
            
            # Complete any ongoing provider down intervals
            for prov, interval in provider_down_intervals.items():
                if interval["end"] is None:  # Provider still down at end
                    interval["end"] = nTimesteps
                    event = SimulationEvent(
                        t=interval["start"], 
                        type="PROVIDER_DOWN",
                        provider=prov,
                        duration=interval["end"] - interval["start"] + 1
                    )
                    if EVENT_AGGREGATION:
                        emit_event_async(current_run_id, event)

        # --- Attack meta extraction --------------------------------------
        attack_happened = nMalAgents > 0 and np.any(attack_decisions == 1)
        if attack_happened:
            attack_ts = np.flatnonzero(attack_decisions)
            target_provider = targets[attack_ts[0]]

            misinfo_ts  = np.flatnonzero(((attack_methods == 1) | (attack_methods == 2)) & attack_decisions)
            cyber_ts    = np.flatnonzero(((attack_methods == 0) | (attack_methods == 2)) & attack_decisions)
            combined_ts = np.flatnonzero((attack_methods == 2) & attack_decisions)

            misinfo_len, cyber_len, combined_len = len(misinfo_ts), len(cyber_ts), len(combined_ts)
            misinfo_begin = misinfo_ts.min() + 1 if misinfo_len else ""
            cyber_begin   = cyber_ts.min() + 1 if cyber_len else ""
            comb_begin    = combined_ts.min() + 1 if combined_len else ""

            # Malicious reward windows -----------------------------------
            reward_ts = np.append(attack_ts[1:], attack_ts[-1] + (attack_ts[-1] != nTimesteps - 1))
            attackreward    = attack_rewards[reward_ts].sum()
            detectionreward = detection_rewards[reward_ts].sum()
            malreward       = attackreward + detectionreward

            print(
                f"Target: {target_provider + 1} , attack timesteps: {attack_ts} , "
                f"length of attack campaign: {attack_ts.size} , misinfo campaign: {misinfo_ts.tolist()} , "
                f"cyber campaign: {cyber_ts.tolist()}"
            )

            # ── Build events list from ground-truth arrays ───────────────
            typ_map = {0: "CYBER", 1: "MISINFO", 2: "COMBO"}
            last_typ = None
            for t_step in np.flatnonzero(attack_decisions):
                typ = typ_map[int(attack_methods[t_step])]
                if typ != last_typ:
                    events.append({"t": int(t_step) + 1, "type": typ})
                    last_typ = typ
        # ------------------------------------------------------------------

        # Provider‑level metrics -------------------------------------------
        for provider in range(nProviders):
            # Averages after warm‑up
            avguse = np.mean(actions[W + 1:, provider]) / nAgents
            denom_total = np.sum(actions[W + 1:, provider])
            avgreward = 0 if denom_total == 0 else np.sum(rewards[W + 1:, provider]) / denom_total
            posopinion = np.mean(opinionvalues[W + 1:, provider + nProviders])
            negopinion = np.mean(opinionvalues[W + 1:, provider])

            if attack_happened:
                # Slices before / during / after attack of equal length to attack window
                win = attack_ts.size
                before_sl = slice(max(0, attack_ts[0] - win), attack_ts[0])
                during_sl = attack_ts
                after_sl  = slice(attack_ts[-1], min(nTimesteps, attack_ts[-1] + win))

                def _mean_safe(arr):
                    return float(arr.mean()) if arr.size else 0

                avgusebefore = np.mean(actions[before_sl, provider]) / nAgents
                avguseduring = np.mean(actions[during_sl, provider]) / nAgents
                avguseafter  = np.mean(actions[after_sl,  provider]) / nAgents

                avgrewbefore = _mean_safe(rewards[before_sl, provider] / np.where(actions[before_sl, provider] == 0, 1, actions[before_sl, provider]))
                avgrewduring = _mean_safe(rewards[during_sl, provider] / np.where(actions[during_sl, provider] == 0, 1, actions[during_sl, provider]))
                avgrewafter  = _mean_safe(rewards[after_sl,  provider] / np.where(actions[after_sl,  provider] == 0, 1, actions[after_sl,  provider]))

                posopinionbefore = _mean_safe(opinionvalues[before_sl, provider + nProviders])
                posopinionduring = _mean_safe(opinionvalues[during_sl, provider + nProviders])
                posopinionafter  = _mean_safe(opinionvalues[after_sl,  provider + nProviders])
                negopinionbefore = _mean_safe(opinionvalues[before_sl, provider])
                negopinionduring = _mean_safe(opinionvalues[during_sl, provider])
                negopinionafter  = _mean_safe(opinionvalues[after_sl,  provider])

                impactusage = avguse - avguseduring
                impactposop = posopinion - posopinionduring
                impactnegop = abs(negopinion) - abs(negopinionduring)
            else:
                # Placeholders ------------------------------------------------
                (avgusebefore, avguseduring, avguseafter, avgrewbefore, avgrewduring, avgrewafter,
                 posopinionbefore, posopinionduring, posopinionafter, negopinionbefore,
                 negopinionduring, negopinionafter, impactusage, impactposop, impactnegop) = ("",) * 15

            # Row assembly ---------------------------------------------------
            rows.append({
                "regret": avgregret,
                "provider": provider + 1,
                "avguse": avguse,
                "avgreward": avgreward,
                "posopinion": posopinion,
                "negopinion": negopinion,
                "avgusebefore": avgusebefore,
                "avguseduring": avguseduring,
                "avguseafter": avguseafter,
                "avgrewbefore": avgrewbefore,
                "avgrewduring": avgrewduring,
                "avgrewafter": avgrewafter,
                "posopinionbefore": posopinionbefore,
                "posopinionduring": posopinionduring,
                "posopinionafter": posopinionafter,
                "negopinionbefore": negopinionbefore,
                "negopinionduring": negopinionduring,
                "negopinionafter": negopinionafter,
                "target_id": (target_provider + 1) if attack_happened else "",
                "attack_beginning": (attack_ts[0] + 1) if attack_happened else "",
                "misinfo_beginning": misinfo_begin if attack_happened else "",
                "misinfo_length": misinfo_len if attack_happened else "",
                "cyber_beginning": cyber_begin if attack_happened else "",
                "cyber_length": cyber_len if attack_happened else "",
                "combined_beginning": comb_begin if attack_happened else "",
                "combined_length": combined_len if attack_happened else "",
                "impactusage": impactusage,
                "impactposop": impactposop,
                "impactnegop": impactnegop,
                "attackreward": attackreward if attack_happened else "",
                "detectionreward": detectionreward if attack_happened else "",
                "malreward": malreward if attack_happened else "",
                # parameters – keep identical names -------------------------
                "alpha": alpha,
                "delta": delta,
                "epsilon": epsilon,
                "eta": eta,
                "theta": theta,
                "kappa": kappa,
                "Lambda": Lambda[provider],
                "lambda": lambda_arr[provider],
                "xi": xi,
                "rho": rho,
                "tau": tau,
                "Upsilon": Upsilon[provider],
                "upsilon": upsilon_[provider],
            })
    # ───────────────────────────────────────────────────────────────────────

    # Summary CSV is tiny → easy
    data_df = pd.DataFrame.from_records(rows, columns=COLUMNS)
    
    # ── SAVE PROVIDER-LEVEL SUMMARY (unchanged filename) ───────────────────
    summary_path = os.path.join(OUTPUT_DATA_DIR, f"experiment{experiment_id}_summary.csv")
    data_df.to_csv(summary_path, mode="a", header=False, index=False, encoding="utf-8")

    # ---- persist header so front-end sees meta + attack timeline ---------
    jsonl.close()          # flush the frame stream first


    # TEST_FRAMES = FRAMES_JSONL.replace('.jsonl.gz', '_small.jsonl.gz')
    # with gzip.open(FRAMES_JSONL, 'rt') as f_in:
    #     with gzip.open(TEST_FRAMES, 'wt') as f_out:
    #         for i, line in enumerate(f_in):
    #             if i < 10000:  # Only first 10k lines
    #                 f_out.write(line)
    #             else:
    #                 break
    # print(f"Created small test file: {TEST_FRAMES}")

    header_json = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nAgents":    nAgents,
            "nTimesteps": nTimesteps,
            # add whatever params you want surfaced in the UI ↓
        },
        "summary_csv": os.path.basename(SUMMARY_CSV),
        "frames_gzip": os.path.basename(FRAMES_JSONL),
        "events": events,          # ← now actually persisted
        "links": [{"id": edge_id(s, t), "source": s, "target": t, "layer": "info", "phase": "NONE", "dir": "↔", "type": typ} for s, t, typ in links]
    }
    # std-lib json is fine (events are plain ints/strs)
    with open(HEADER_JSON, "w", encoding="utf-8") as fh:
        json.dump(header_json, fh)

    timer.stop()

    # ----------------------------------------------------------------------
    #  📊  Visualisations (identical logic, but head‑less safe) 
    # ----------------------------------------------------------------------

    plt.rcParams["figure.figsize"] = (20, 4)

    # Plot 1 – provider usage over time ------------------------------------
    A_over_time = (actions / nExperiments) * 100 / nAgents
    for prov in range(nProviders):
        ts = np.arange(1, len(A_over_time) + 1)
        plt.plot(ts, A_over_time[:, prov], "-", label=f"Provider {prov + 1}")
    plt.xlabel("Time steps")
    plt.ylabel("Average provider usage (%)")
    plt.xlim(1, nTimesteps)
    plt.ylim(1, 100)
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_fig:
        plt.savefig(os.path.join(output_dir, "actions.pdf"), dpi=500, bbox_inches="tight")
    _maybe_show()

    # Plot 2 – average reward per provider ---------------------------------
    aveR_A = rewards / np.where(actions == 0, 1, actions)
    for prov in range(nProviders):
        ts = np.arange(1, len(aveR_A) + 1)
        plt.plot(ts, aveR_A[:, prov], ".", label=f"Provider {prov + 1}")
    plt.xlabel("Time steps")
    plt.ylabel("Average reward per service provider")
    plt.xlim(1, nTimesteps)
    plt.ylim(-0.01, 1.01)
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_fig:
        plt.savefig(os.path.join(output_dir, "rewards.pdf"), dpi=500, bbox_inches="tight")
    _maybe_show()

    # Plot 3 – opinion dynamics -------------------------------------------
    for opinion in range(nProviders * 2):
        ts = np.arange(1, len(opinionvalues) + 1)
        style = "-" if opinion >= nProviders else "--"
        color = ["tab:blue", "tab:orange", "tab:green"][opinion % nProviders]
        sign  = "+" if opinion >= nProviders else "-"
        label = f"Provider {opinion % nProviders + 1} {sign}"
        plt.plot(ts, opinionvalues[:, opinion], style, color=color, label=label)
    plt.xlabel("Time steps")
    plt.ylabel("Average evaluation of opinion value per service provider")
    plt.xlim(1, nTimesteps)
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_fig:
        plt.savefig(os.path.join(output_dir, "opinionvalues.pdf"), dpi=500, bbox_inches="tight")
    _maybe_show()

    # Plot 4 – malicious rewards ------------------------------------------
    ts = np.arange(1, len(attack_rewards) + 1)
    plt.plot(ts, attack_rewards, ":", label="Attack rewards")
    plt.plot(ts, detection_rewards, ":", label="Detection rewards")
    plt.xlabel("Time steps")
    plt.ylabel("Malicious users' rewards")
    plt.xlim(1, nTimesteps)
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_fig:
        plt.savefig(os.path.join(output_dir, "malrewards.pdf"), dpi=500, bbox_inches="tight")
    _maybe_show()

    print("✅  Run {}/{} finished in {:.2f} s.\n"
          "   ↳ summary CSV   → {}\n"
          "   ↳ frames JSONL  → {}\n"
          "   ↳ header JSON   → {}"
          .format(_run_idx + 1, run, timer.elapsed,
                  SUMMARY_CSV, FRAMES_JSONL, HEADER_JSON))
    
    # End live streaming run if available
    if LIVE_STREAMING and current_run_id:
        try:
            requests.post(f"http://localhost:8000/api/v1/runs/{current_run_id}/end", timeout=5)
        except Exception as e:
            print(f"Failed to end run: {e}")
        print(f"   ↳ live run ID   → {current_run_id}")