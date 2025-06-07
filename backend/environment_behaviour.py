#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random

np.random.seed(0)

class EnvironmentBehaviour:
    """Environment behaviour"""

    def __init__(self, agents, regagents, malagents, providers, neighbours, network, nTimesteps):
        """Initialise enviornment's behaviour"""
        self.agents = agents
        self.regagents = regagents
        self.malagents = malagents
        self.providers = providers
        self.neighbours = neighbours
        self.network = network
        self.nTimesteps = nTimesteps
        self.states = np.zeros((self.nTimesteps+1, 3, len(self.providers)), dtype = int)
        self.regret = np.zeros(self.nTimesteps+1, dtype = int)
        self.cyberattack_reward = np.zeros(self.nTimesteps+1, dtype = float)
        self.misinfo_reward = np.zeros(self.nTimesteps+1, dtype = float)
        self.cyber_detection_reward = np.zeros(self.nTimesteps+1, dtype = float)
        self.misinfo_detection_reward = np.zeros(self.nTimesteps+1, dtype = float)
        self.cyber_contact, self.misinfo_contact = [], []
        self.targets = np.full(self.nTimesteps+1 , fill_value = -1)
        self.attack_methods = np.full(self.nTimesteps+1, fill_value = -1)

    def calculate_malreward(self, eta, theta, timestep):
        """
        attack_reward : tuple (cyberattack reward, misinfo reward)
        detection_reward : tuple (cyberattack detection reward, misinformation detection reward)
        """
        if self.attack_methods[timestep-1]==0: # calculate reward for a cyberattack
            cyberattack_reward = self.calculate_cyberattack_reward(timestep)
            cyber_detection_reward = self.calculate_cyber_detection_reward(theta, timestep)
            return (cyberattack_reward,0), (cyber_detection_reward,0)
        elif self.attack_methods[timestep-1]==1: # calculate reward for a misinformation campaign
            misinfo_reward = self.calculate_misinfo_reward(timestep)
            misinfo_detection_reward = self.calculate_misinfo_detection_reward(eta, timestep)
            return (0,misinfo_reward), (0,misinfo_detection_reward)
        elif self.attack_methods[timestep-1]==2: # calculate reward for a coordinated attack
            cyberattack_reward = self.calculate_cyberattack_reward(timestep)
            misinfo_reward = self.calculate_misinfo_reward(timestep)
            cyber_detection_reward = self.calculate_cyber_detection_reward(theta, timestep)
            misinfo_detection_reward = self.calculate_misinfo_detection_reward(eta, timestep)
            return (cyberattack_reward, misinfo_reward), (cyber_detection_reward, misinfo_detection_reward)
        else:
            return (0,0), (0,0)

    def calculate_cyberattack_reward(self, timestep):
        """
        cyberattack_reward : float
            Reward is the percentage of users that take service from the provider at the
            timestep of an attack
        """
        # Calculate cyberattack reward
        self.cyberattack_reward[timestep] = (self.states[timestep-1, 1, self.targets[timestep-1]])/len(self.agents)
        if np.all(self.states[:,2,:]==0): 
            # Give negative reward if no actual cyberattack has started,
            # but they observed somebody considering the possibility of a cyberattack
            self.cyberattack_reward[timestep] = -self.cyberattack_reward[timestep]
        elif np.any(self.states[timestep-1,2,:]==1): 
            # Give positive reward if attack happened previous time step
            self.cyberattack_reward[timestep] = self.cyberattack_reward[timestep]
        else: 
            # Give no reward after the end of an attack campaign
            self.cyberattack_reward[timestep] = 0
        return self.cyberattack_reward[timestep]

    def calculate_misinfo_reward(self, timestep):
        """
        misinfo_reward : float
            Reward is the percentage of users who asked information about the target
            from a malicious agent
        """
        # Calculate misinformation reward
        misinforeward_list = []
        for agent in self.regagents:
            if self.network.nodes[agent]["zeta"][timestep-1] in self.malagents:
                # Include agent in list if one was lied by a malicious agent
                misinforeward_list.append(1)
            else:
                misinforeward_list.append(0)
        self.misinfo_reward[timestep] = np.sum(misinforeward_list)/len(self.regagents)
        if np.all(self.states[:,2,:]==0):
            # Give negative reward if no actual misinfo has not started,
            # but they observe somebody considering the possibility of a misinformation attack
            self.misinfo_reward[timestep] = -self.misinfo_reward[timestep]
        elif np.any(self.states[timestep-1,2,:]==1): 
            # Give positive reward if attack happened previous time step
            self.misinfo_reward[timestep] = self.misinfo_reward[timestep]
        else: 
            # Give no reward after the end of an attack campaign
            self.misinfo_reward[timestep] = 0
        return self.misinfo_reward[timestep]
    
    def calculate_cyber_detection_reward(self, theta, timestep):
        """
        detection_reward : float
            Before an attack, reward value is 0. 
            During an attack, the reward, that is the percentage of agents 
            who suffered from the attack, starts to decrease.
        """
        if np.all(self.states[:,2,:]==0): 
            # Give no negative reward if attack has not happened
            self.cyber_detection_reward[timestep] = 0
        elif np.any(self.states[timestep-1,2,:]==1): 
            # Give negative reward if cyberattack happened previous time step
            cyber_contact = []
            for agent in self.regagents:
                if agent not in self.cyber_contact and self.network.nodes[agent]["a"][timestep-1] == self.targets[timestep-1] and \
                  random.random() <= theta:
                    # Include agent in contact list ('has detected') with certain probability
                    self.cyber_contact.append(agent)
                    cyber_contact.append(1)
                else:
                    cyber_contact.append(0)
            self.cyber_detection_reward[timestep] = self.cyber_detection_reward[timestep-1]-(np.sum(cyber_contact)/(len(self.regagents)-len(self.cyber_contact)))
        else: 
            # Give no negative reward after the end of an attack campaign
            self.cyber_detection_reward[timestep] = 0
        return self.cyber_detection_reward[timestep]

    def calculate_misinfo_detection_reward(self, eta, timestep):
        """
        detection_reward : float
            Before an attack, reward value is 0. 
            During an attack, the reward that is the percentage of agents who were 
            in contact with a malicious agent, starts to decrease.
        """
        if np.all(self.states[:,2,:]==0): 
            # Give no negative reward if attack has not happened
            self.misinfo_detection_reward[timestep] = 0
        elif np.any(self.states[timestep-1,2,:]==1): 
            # Give negative reward if misinformation was spread previous time step
            misinfo_contact = []
            for agent in self.regagents:
                if agent not in self.misinfo_contact and (self.network.nodes[agent]["zeta"][timestep-1] in self.malagents\
                    or self.network.nodes[agent]["zeta"][timestep-1] in self.misinfo_contact)\
                    and random.random() <= eta:
                    # Include agent in contact list ('has detected') with certain probability
                    self.misinfo_contact.append(agent)
                    misinfo_contact.append(1)
                else:
                    misinfo_contact.append(0)

            self.misinfo_detection_reward[timestep] = self.misinfo_detection_reward[timestep-1]-(np.sum(misinfo_contact)/(len(self.regagents)-len(self.misinfo_contact)))  
        else: 
            # Give no negative reward after the end of an attack campaign
            self.misinfo_detection_reward[timestep] = 0
        return self.misinfo_detection_reward[timestep]
    
    def show_state(self, timestep):
        """
        state : array
            The current state includeing central service states, number of customers, 
            and actions taken by malicious user's neighbours.
        """
        self.states[timestep,0] = self.states[timestep-1,0]
        self.states[timestep,1] = self.states[timestep-1,1]
        self.states[timestep,2] = self.states[timestep-1,2]

    def collect_data(self, countusers, center, attack_decision, target, attack_method, timestep):
        # Collect data to show as state in the next time step
        self.states[timestep,0] = center[timestep]
        self.states[timestep,1] = countusers[timestep]
        if attack_decision == 1:
            self.states[timestep,2,target] = 1
        else:
            self.states[timestep,2,:] = 0
        self.targets[timestep] = target
        self.attack_methods[timestep] = attack_method
    
    def show_action_values(self, actionValues, timestep):
        """
        avg_action : float
            The average action value of all agents 
        """
        
        action_dyn = np.zeros((len(self.agents), len(self.providers)), dtype = float)
        avg_action = np.zeros(len(self.providers), dtype = float)

        for agent in self.agents:
            action_dyn[agent] = actionValues[timestep,agent]
        
        for action in range(len(self.providers)):
            avg_action[action] = sum(action_dyn[:,action])/len(self.agents)
        return avg_action


    def show_opinion_dynamics(self, opinionValues, timestep):
        """
        avg_opinion : float
            The average opinion value of all agents 
        """
        avg_opinion = np.zeros(len(self.providers)*2, dtype = float)
        timestep_opinionValues = opinionValues[timestep]

        for opinion in range(len(self.providers)*2):
            avg_opinion[opinion] = sum(timestep_opinionValues[:,opinion])/len(self.agents)
        return avg_opinion
    
    def calculate_regret(self, endpoint, timestep):
        """
        optimal_action a*: int
            Provider whose service had the highest reward value
        optimal_value V*: (binary) int
            The best reward value from all actions
        regret : int
            The opportinity loss
        """
        for agent in self.agents:
            optimal_action = np.argmax(endpoint[agent,timestep])
            optimal_value = endpoint[agent,timestep,optimal_action]
            regret = optimal_value - self.network.nodes[agent]["r(a)"][timestep]
            self.regret[timestep] += regret
        return self.regret[timestep]
