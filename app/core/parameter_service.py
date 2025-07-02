"""Service for parsing and processing simulation parameters from JSON input."""
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


class SimulationConfig:
    """Configuration object holding all simulation parameters with descriptive names."""
    
    def __init__(self):
        # Experiment settings
        self.scenarios: List[str] = []
        self.num_timesteps: int = 3000
        self.window_size: int = 10
        self.timestep: int = 1  # Current timestep counter starting value
        
        # Agent configuration
        self.num_agents: int = 248
        self.num_providers: int = 3
        self.num_malicious_agents: int = 0
        self.agent_types: Dict[str, Dict[str, Any]] = {}
        
        # Platform reliability
        self.endpoint_failure_rate: float = 0.03
        self.endpoint_recovery_rate: float = 0.25
        self.core_failure_rate: float = 0.01
        self.core_recovery_rate: float = 0.30
        
        # User behavior
        self.experience_weight: float = 0.80
        self.satisfaction_threshold: float = 0.80
        
        # Threat awareness
        self.cyber_detection_rate: float = 0.80
        self.misinfo_detection_rate: float = 0.10
        
        # Learning parameters
        self.learning_rate: float = 0.05
        self.exploration_rate: float = 0.10
        self.initial_q_value: float = 100.0
        
        # Network topology
        self.num_neighbours: int = 6
        self.rewiring_probability: float = 0.01
        self.networks: Dict[str, Dict[str, Any]] = {}
        
        # Provider-specific overrides
        self.provider_overrides: Dict[int, Dict[str, float]] = {}


class ParameterService:
    """Service for parsing JSON parameters and creating simulation configuration."""
    
    @staticmethod
    def load_defaults() -> SimulationConfig:
        """Load default configuration matching current setup_values.py."""
        config = SimulationConfig()
        
        # Set default agent types
        config.agent_types = {
            "regular": {
                "count": 235,
                "parameters": {
                    "learning_rate": 0.05,
                    "exploration_rate": 0.10
                }
            },
            "malicious": {
                "count": 13,
                "parameters": {
                    "learning_rate": 0.05,
                    "exploration_rate": 0.10,
                    "initial_q_value": 100
                }
            }
        }
        
        # Set default network
        config.networks = {
            "social": {
                "type": "watts-strogatz",
                "parameters": {
                    "num_neighbours": 6,
                    "rewiring_probability": 0.01
                }
            }
        }
        
        # Default scenario
        config.scenarios = ["coordinatedAttack"]
        
        return config
    
    @staticmethod
    def parse_parameters(params: Optional[Dict[str, Any]]) -> SimulationConfig:
        """Parse JSON parameters and merge with defaults."""
        config = ParameterService.load_defaults()
        
        if not params:
            return config
        
        # Parse experiment settings
        if "experiment" in params:
            exp = params["experiment"]
            if "scenarios" in exp:
                config.scenarios = exp["scenarios"]
            if "timesteps" in exp:
                config.num_timesteps = exp["timesteps"]
            if "windowSize" in exp:
                config.window_size = exp["windowSize"]
        
        # Parse agent configuration
        if "agents" in params:
            config.agent_types = params["agents"]
            # Calculate total agents
            config.num_agents = sum(
                agent_def.get("count", 0) 
                for agent_def in config.agent_types.values()
            )
            # Calculate malicious agents
            config.num_malicious_agents = config.agent_types.get("malicious", {}).get("count", 0)
        
        # Parse networks
        if "networks" in params:
            config.networks = params["networks"]
            # Extract primary network parameters
            primary_network = next(iter(config.networks.values()), {})
            if primary_network.get("type") == "watts-strogatz":
                net_params = primary_network.get("parameters", {})
                if "numberOfNeighbours" in net_params:
                    config.num_neighbours = net_params["numberOfNeighbours"]
                if "rewiringProbability" in net_params:
                    config.rewiring_probability = net_params["rewiringProbability"]
        
        # Parse provider configuration
        if "providers" in params:
            prov = params["providers"]
            if "config" in prov:
                prov_config = prov["config"]
                if "count" in prov_config:
                    config.num_providers = prov_config["count"]
                if "defaultEndpointDowntime" in prov_config:
                    config.endpoint_failure_rate = prov_config["defaultEndpointDowntime"]
                if "defaultEndpointRecovery" in prov_config:
                    config.endpoint_recovery_rate = prov_config["defaultEndpointRecovery"]
                if "defaultCoreDowntime" in prov_config:
                    config.core_failure_rate = prov_config["defaultCoreDowntime"]
                if "defaultCoreRecovery" in prov_config:
                    config.core_recovery_rate = prov_config["defaultCoreRecovery"]
            
            # Parse provider-specific overrides
            if "instances" in prov:
                for instance in prov["instances"]:
                    provider_id = instance["id"]
                    overrides = {}
                    if "endpointDowntime" in instance:
                        overrides["endpoint_failure_rate"] = instance["endpointDowntime"]
                    if "endpointRecovery" in instance:
                        overrides["endpoint_recovery_rate"] = instance["endpointRecovery"]
                    if "coreDowntime" in instance:
                        overrides["core_failure_rate"] = instance["coreDowntime"]
                    if "coreRecovery" in instance:
                        overrides["core_recovery_rate"] = instance["coreRecovery"]
                    config.provider_overrides[provider_id] = overrides
        
        # Parse global behavior
        if "globalBehavior" in params:
            behavior = params["globalBehavior"]
            if "experienceWeight" in behavior:
                config.experience_weight = behavior["experienceWeight"]
            if "satisfactionThreshold" in behavior:
                config.satisfaction_threshold = behavior["satisfactionThreshold"]
            if "cyberDetectionRate" in behavior:
                config.cyber_detection_rate = behavior["cyberDetectionRate"]
            if "misinformationDetectionRate" in behavior:
                config.misinfo_detection_rate = behavior["misinformationDetectionRate"]
        
        # Extract learning parameters from agent types if available
        if "regular" in config.agent_types:
            reg_params = config.agent_types["regular"].get("parameters", {})
            if "learningSpeed" in reg_params:
                config.learning_rate = reg_params["learningSpeed"]
            if "exploreBias" in reg_params:
                config.exploration_rate = reg_params["exploreBias"]
        
        if "malicious" in config.agent_types:
            mal_params = config.agent_types["malicious"].get("parameters", {})
            if "positiveInitialisation" in mal_params:
                config.initial_q_value = mal_params["positiveInitialisation"]
        
        return config
    
    @staticmethod
    def get_provider_params(config: SimulationConfig, provider_id: int) -> Dict[str, float]:
        """Get parameters for a specific provider, applying overrides if present."""
        params = {
            "endpoint_failure_rate": config.endpoint_failure_rate,
            "endpoint_recovery_rate": config.endpoint_recovery_rate,
            "core_failure_rate": config.core_failure_rate,
            "core_recovery_rate": config.core_recovery_rate
        }
        
        # Apply provider-specific overrides
        if provider_id in config.provider_overrides:
            params.update(config.provider_overrides[provider_id])
        
        return params
    
    @staticmethod
    def determine_experiment_id(config: SimulationConfig) -> int:
        """Determine experiment ID based on active scenarios."""
        if "cyberAttack" in config.scenarios and "misinformation" not in config.scenarios:
            return 1
        elif "misinformation" in config.scenarios and "cyberAttack" not in config.scenarios:
            return 2
        elif "coordinatedAttack" in config.scenarios:
            return 3
        else:
            return 0  # No attack
    
    @staticmethod
    def validate_config(config: SimulationConfig) -> List[str]:
        """Validate configuration and return list of any errors."""
        errors = []
        
        if config.num_agents < 1:
            errors.append("Number of agents must be at least 1")
        
        if config.num_providers < 1:
            errors.append("Number of providers must be at least 1")
        
        if config.num_timesteps < 1:
            errors.append("Number of timesteps must be at least 1")
        
        if not (0 <= config.learning_rate <= 1):
            errors.append("Learning rate must be between 0 and 1")
        
        if not (0 <= config.exploration_rate <= 1):
            errors.append("Exploration rate must be between 0 and 1")
        
        if not (0 <= config.cyber_detection_rate <= 1):
            errors.append("Cyber detection rate must be between 0 and 1")
        
        if not (0 <= config.misinfo_detection_rate <= 1):
            errors.append("Misinformation detection rate must be between 0 and 1")
        
        if not config.agent_types:
            errors.append("At least one agent type must be defined")
        
        if not config.networks:
            errors.append("At least one network must be defined")
        
        return errors