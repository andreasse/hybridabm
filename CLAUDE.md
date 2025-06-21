# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hybrid threat modeling system that uses **Agent-Based Modeling (ABM)** combined with **Reinforcement Learning (RL)** to simulate coordinated cyberattacks and misinformation campaigns. The system models interactions between malicious agents, regular agents, and service providers in a cyber-physical-social system (CPSS).

## Running the Code

**Main Command:**
```bash
python run.py
```

**Requirements:**
- Python 3.10 or higher
- Install dependencies: `pip install -r requirements.txt`

## Core Architecture

### System Components (Execution Order per Timestep)
1. **Environment** (`environment.py`) - Creates network topology, manages agents/providers
2. **Malicious Agents** (`maliciousagent_behaviour.py`) - Execute attacks using Q-learning
3. **Service Providers** (`serviceprovider_behaviour.py`) - Two-state Markov model for availability
4. **Regular Agents** (`agent_behaviour.py`) - Select providers and express opinions via RL
5. **Environment Behavior** (`environment_behaviour.py`) - Calculate rewards and collect data

### Key Design Patterns

**Reinforcement Learning Integration:**
- All agents use Q-learning with separate Q-tables for actions and opinions
- Malicious agents learn attack effectiveness through environmental rewards
- Regular agents adapt both service selection and opinion expression

**Multi-Agent Coordination:**
- Social network enables information spread via Watts-Strogatz topology
- Coordinated attacks combine cyber and misinformation vectors
- Opinion dynamics influence service provider selection

### Configuration System (`setup_values.py`)

**Experiment Types** (controlled by boolean flags):
- `cyberattack = True/False` - Independent cyber attacks only
- `misinformation = True/False` - Independent misinformation campaigns only  
- `coordinated_attack = True/False` - Coordinated hybrid attacks
- All `False` = Baseline (no malicious agents)

**Key Parameters:**
- `nAgents` - Total agent population
- `nProviders` - Number of service providers
- Network topology: `kappa` (rewiring probability), `rho` (connectivity)
- Learning rates: `alpha_a` (actions), `alpha_o` (opinions)
- Exploration: `epsilon` parameters

## Data Output Structure

**Files Generated:**
- `experiment3_summary.csv` - Aggregated provider-level metrics
- `frames_exp3_*.jsonl.gz` - Timestep-level detailed data (compressed)
- `header_exp3_*.json` - Experiment metadata and attack timelines

**Key Metrics Tracked:**
- Service provider usage patterns (before/during/after attacks)
- Opinion dynamics (positive/negative spread)
- Attack coordination and effectiveness
- Agent regret and adaptation patterns

## Logging Architecture

### Core Logging Principles

**File-Based Logging Only:**
- **NEVER use print statements** - Simulation generates hundreds of step requests per second
- **Use logging.FileHandler exclusively** - Structured log files for debugging complex interactions
- **Frontend logs** - Available in browser console for WebSocket connection status

### Log File Structure

**Primary Log Files:**
- **`simulation_debug.log`** - Simulation timesteps, event generation, background event processing
- **`event_aggregator.log`** - EventAggregator operations, simulation events, WebSocket client management
- **`websocket_debug.log`** - Dedicated WebSocket connection lifecycle, registration, and errors

### Critical Logging Issue & Resolution

**Problem Discovered**: Logger conflicts between components caused missing debug information

**The Issue**:
```python
# EventAggregator creates logger with overwrite mode
handler = logging.FileHandler('event_aggregator.log', mode='w')

# WebSocket endpoint tried to append to same logger
handler = logging.FileHandler('event_aggregator.log', mode='a')
```

**Result**: WebSocket connection logs were being lost or overwritten, making debugging impossible

**Solution Applied**:
```python
# Separate loggers with dedicated files
ws_logger = logging.getLogger('websocket_debug')
ws_handler = logging.FileHandler('websocket_debug.log', mode='w')

event_logger = logging.getLogger('event_aggregator')
event_handler = logging.FileHandler('event_aggregator.log', mode='w')
```

### Logging Best Practices

**When to Create Separate Log Files**:
- **Different lifecycles** - EventAggregator persists, WebSocket connections are transient
- **Different components** - Simulation engine vs FastAPI server vs event streaming
- **High volume vs debugging** - Separate noisy operations from critical debugging info
- **Mode conflicts** - When components need different file modes (overwrite vs append)

**Logger Configuration Pattern**:
```python
# Create dedicated logger
logger = logging.getLogger('component_name')
logger.setLevel(logging.DEBUG)

# Clear existing handlers to avoid conflicts
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add dedicated file handler
handler = logging.FileHandler('component_debug.log', mode='w')
handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
logger.addHandler(handler)
```

**Critical Lesson**: Complex systems with multiple components require **isolated logging channels** to prevent interference and ensure debuggability.

## Development Workflow

**Simulation Loop Structure:**
1. Environment displays state
2. Calculate malicious agent rewards (timestep 2+)
3. Malicious agents update policies and execute attacks
4. Service providers transition states
5. Regular agents select actions and receive rewards
6. Opinion expression and social influence
7. Data collection and regret calculation

**Extension Points:**
- Add new attack types in `maliciousagent_behaviour.py`
- Modify network topology in `environment.py:form_network()`
- Extend metrics collection in `environment_behaviour.py`
- Create new agent types following existing behavioral patterns

## Important Implementation Notes

- **Timestep Dependencies:** Malicious agents act before regular agents to establish cause-effect relationships
- **State Management:** Service providers use central vs endpoint states for different attack impacts  
- **Social Influence:** Opinion updates occur after action selection to model information cascades
- **Data Efficiency:** Uses orjson for streaming large simulation datasets
- **Network Effects:** Watts-Strogatz topology balances clustering and path length for realistic information spread