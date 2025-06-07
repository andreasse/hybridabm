#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from setup_values import nProviders, nAgents, nMalAgents, cyberattack, misinformation, coordinated_attack,\
    Upsilon, upsilon_, Lambda, lambda_, kappa, rho, delta, alpha, epsilon, tau,\
    xi, eta, theta, W
from environment import Environment
from environment_behaviour import EnvironmentBehaviour
from agent_behaviour import AgentsBehaviour
from serviceprovider_behaviour import ServiceProvidersBehaviour
from maliciousagent_behaviour import MaliciousAgentsBehaviour

def experiment(timestep, nTimesteps):
    env = Environment(nAgents, nMalAgents, nProviders, nTimesteps)
    network, neighbours = env.form_network(kappa, rho)
    links = [(str(min(u,v)), str(max(u,v)), "info")            # ↓ ensures ("4","10") and ("10","4") don't both sneak in
             for u, v in network.edges]
    final_provider_map = set()
    agents, regagents, malagents = env.create_agents()
    providers = env.create_providers()
    env.create_attributes()
    
    env_behaviour = EnvironmentBehaviour(agents, regagents, malagents, providers, neighbours, network, nTimesteps)
    reg_behaviour = AgentsBehaviour(network, providers)
    serv_behaviour = ServiceProvidersBehaviour(providers, agents, Upsilon, upsilon_, Lambda, lambda_, nTimesteps)
    mal_behaviour = MaliciousAgentsBehaviour(network, malagents, providers, xi, nTimesteps)
    
    countUsers = np.zeros((nTimesteps+1, nProviders), dtype=int)
    countRewards = np.zeros((nTimesteps+1, nProviders), dtype = int)
    opinionValues = np.zeros((nTimesteps+1, len(agents), 2*len(providers)), dtype = float)
    actions, rewards, regrets, opinionvalues, attack_decisions, targets, attack_methods,\
         attack_rewards, detection_rewards = [], [], [], [], [], [], [], [], []
    

    for step in range(nTimesteps):
        # Environment
        env_behaviour.show_state(timestep)
        attack_reward, detection_reward = env_behaviour.calculate_malreward(eta, theta, timestep)
        # Malicious agents
        mal_behaviour.receive_malreward(attack_reward, detection_reward, timestep)
        mal_behaviour.update_malQ(alpha, W, timestep)
        attack_decision = mal_behaviour.select_action(cyberattack, misinformation, coordinated_attack, W, timestep)
        target = mal_behaviour.select_target(attack_decision, cyberattack, misinformation, W, timestep)
        attack_method = mal_behaviour.select_attack_method(attack_decision, target, epsilon, cyberattack, misinformation, coordinated_attack,\
             W, timestep)
        
        # Service providers
        for provider in providers:
            center = serv_behaviour.central_state(provider, target, attack_method, attack_decision, timestep)
            endpoint = serv_behaviour.endpoint_level(provider, center, timestep)
        # All agents   
        for agent in agents:
            action = reg_behaviour.select_action(agent, epsilon, tau, timestep)
            final_provider_map.add((agent, action))
            reward = reg_behaviour.receive_reward(agent, action, endpoint, timestep)
            reg_behaviour.update_Q(agent, action, reward, alpha)
            opinion = reg_behaviour.express_opinion(agent, epsilon, timestep)
            neighbour = reg_behaviour.find_neighbour(agent, timestep)
            feedback = reg_behaviour.ask_info(neighbour, malagents, mal_behaviour, opinion, delta,
                 attack_decision, attack_method)
            opinion_values = reg_behaviour.update_opinion_value(agent, opinion, feedback, alpha)
            # Collect agent specific information
            countUsers[timestep,action]+=1
            countRewards[timestep,action]+=reward
            opinionValues[timestep,agent]=opinion_values
        # Environment collects data from this state
        env_behaviour.collect_data(countUsers, center, attack_decision, target, attack_method, timestep)
        avg_opinions = env_behaviour.show_opinion_dynamics(opinionValues,timestep)        
        regret = env_behaviour.calculate_regret(endpoint, timestep)
            
        # Collect data for further analysis
        actions.append(countUsers[timestep])
        rewards.append(countRewards[timestep])
        opinionvalues.append(avg_opinions)
        regrets.append(regret)
        attack_decisions.append(attack_decision)
        targets.append(target)
        attack_methods.append(attack_method)
        attack_rewards.append(sum(attack_reward))
        detection_rewards.append(sum(detection_reward))
        timestep+=1
    
    # Add cyber edges after simulation
    links.extend([(str(a), f"prov_{p}", "cyber") for a, p in final_provider_map])
    
    return np.array(actions), np.array(rewards), np.array(opinionvalues), np.array(regrets), \
        np.array(attack_decisions), np.array(targets), np.array(attack_methods), np.array(attack_rewards), np.array(detection_rewards), links
    
