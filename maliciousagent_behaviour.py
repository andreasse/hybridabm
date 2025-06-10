#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

np.random.seed(0)

class MaliciousAgentsBehaviour:
    """Malicious agents' behaviour"""

    def __init__(self, network, malagents, providers, xi, nTimesteps):
        """ Describes the behaviour of malicious users"""
        self.network = network
        self.malagents = malagents
        self.providers = providers
        self.noattack = len(self.providers)
        self.nTimesteps = nTimesteps
        self.attack_decisions = np.zeros(self.nTimesteps+1, dtype=int)
        self.targets = np.full(self.nTimesteps+1 , fill_value = -1)
        self.attack_methods = np.full(self.nTimesteps+1, fill_value = -1)
        self.attack_campaign = []
        self.malrewards = np.zeros((self.nTimesteps+1, 2,2), dtype=float)
        self.malQ = np.full((len(providers), 2*2), fill_value = [0,0,xi,xi], dtype = float) 
        
    def select_action(self, cyberattack, misinformation, coordinated_attack, W, timestep):
        """
        attack_decision a_m: int
            Attack decision is either to attack (0) or not (1)
        """
        # Only attack if there are any malicious agents in the network and system has warmed up
        if len(self.malagents) == 0 or W >= timestep: 
            attack_decision = 0 # decide not to attack
        elif ((np.all(self.attack_decisions==0) and cyberattack and np.any(self.malQ[:,0:1] >= np.max(self.malQ[:,2:3])))\
            or (np.all(self.attack_decisions==0) and misinformation and np.any(self.malQ[:,1:2] >= np.max(self.malQ[:,3:4])))\
            or (np.all(self.attack_decisions==0) and coordinated_attack and np.any(self.malQ[:,0:2] >= np.max(self.malQ[:,2:4])))):
            attack_decision = 1 # start attack campaign
        elif ((self.attack_decisions[timestep-1] == 1 and cyberattack and np.any(self.malQ[self.targets[np.where(self.attack_decisions==1)[0][0]],0:1] \
            >= np.max(self.malQ[self.targets[np.where(self.attack_decisions==1)[0][0]],2:3])))\
            or (self.attack_decisions[timestep-1] == 1 and misinformation and np.any(self.malQ[self.targets[np.where(self.attack_decisions==1)[0][0]],1:2] \
            >= np.max(self.malQ[self.targets[np.where(self.attack_decisions==1)[0][0]],3:4])))\
            or (self.attack_decisions[timestep-1] == 1 and coordinated_attack and np.any(self.malQ[self.targets[np.where(self.attack_decisions==1)[0][0]],0:2] \
            >= np.max(self.malQ[self.targets[np.where(self.attack_decisions==1)[0][0]],2:4])))):
            attack_decision = 1 # continue with an attack campaign
        elif np.all(self.attack_decisions!=0) and self.attack_decisions[timestep-1] == 0 and coordinated_attack and\
            (0 not in self.attack_campaign and \
                self.malQ[self.targets[np.where(self.attack_decisions==1)[0][0]],0] >= np.max(self.malQ[:,2:4]) or
            1 not in self.attack_campaign and \
                self.malQ[self.targets[np.where(self.attack_decisions==1)[0][0]],1] >= np.max(self.malQ[:,2:4])):
            attack_decision = 1 # start new attack campaign
        else:
            attack_decision = 0 # decide not to attack
        self.attack_decisions[timestep] = attack_decision
        return attack_decision

    def select_target(self, attack_decision, cyberattack, misinformation, W, timestep):
        """
        target chi_m: int
            Service provider id who is attacked or whose behaviour is observed
        """
        # Only attack if there are any malicious agents in the network and system has warmed up
        if len(self.malagents) == 0 or W >= timestep:
            target = -1
        elif attack_decision == 1:
            if np.all(self.attack_decisions[:timestep]==0):
                # Choose target for the first time using greedy policy (i.e, has produced the highest cost)
                if cyberattack:
                    target = np.where(self.malQ[:,2:3] == (self.malQ[:,2:3]).min())[0][0]
                elif misinformation:
                    target = np.where(self.malQ[:,3:4] == (self.malQ[:,3:4]).min())[0][0]
                else:
                    target = np.where(self.malQ[:,2:4] == (self.malQ[:,2:4]).min())[0][0]
            else:
                target = self.targets[np.where(self.attack_decisions==1)[0][0]]
        else: # Observe somebody's behaviour as the attack decision is 'not attack'
            target = np.random.choice(self.providers)
        self.targets[timestep] = target
        return self.targets[timestep]

    def select_attack_method(self, attack_decision, target, eps, cyberattack, misinformation, coordinated_attack, W, timestep):
        """
        attack_method : int
            Attack method is either 'cyber' or 'misinformation', where 0 is cyber and 1 is misinformation
        """
        # Only attack if there are any malicious agents in the network and system has warmed up
        if len(self.malagents) == 0 or W>=timestep:
            attack_method = -1
        elif cyberattack:
            attack_method = 0 # conduct a cyberattack
        elif misinformation:
            attack_method = 1 # conduct a misinformation campaign
        elif attack_decision == 1 and coordinated_attack:
            if np.all(self.attack_decisions[:timestep]==0):
                # Choose attack method for the first time using greedy policy (i.e., has produced the highest cost)
                attack_method = np.where(self.malQ[target,2:4] == (self.malQ[target,2:4]).min())[0][0]
                # ... or choose attack method defined as 'start_attack_method' (future update)
            elif self.attack_decisions[timestep-1]==1:
                # Compare cyberattack versus cyber orbserve and misinfo attack and misinfo observe action values
                if self.malQ[target,0:1][0] > self.malQ[target,2:3][0] and self.malQ[target,1:2][0] > self.malQ[target,3:4][0]:
                    attack_method = 2 # use both attack methods
                elif self.malQ[target,0:1][0] > self.malQ[target,2:3][0]:
                    attack_method = 0 # conduct a cyberattack
                else:
                    attack_method = 1 # conduct a misinformation campaign
                if attack_method != self.attack_methods[timestep-1]:
                    self.attack_campaign.append(self.attack_methods[timestep-1])
            else: # Start with new attack method
                if 0 not in self.attack_campaign:
                    attack_method = 0 # conduct a cyberattack
                else:
                    attack_method = 1  # conduct a misinformation campaign
        else: # Observe somebody's behaviour as the attack decision is 'not attack'
            attack_method = np.random.choice([0,1])
        self.attack_methods[timestep] = attack_method
        return self.attack_methods[timestep]

    def receive_malreward(self, attack_reward, detection_reward, timestep):
        """
        malreward : array
            Reward value for taking action, where array column is 'cyber' (0) and 'misinformation' (1)
            and array row is attack reward (0) and detection reward (1)
        """
        # Collect reward information
        self.malrewards[timestep,0,0] = attack_reward[0]
        self.malrewards[timestep,0,1] = attack_reward[1]
        self.malrewards[timestep,1,0] = detection_reward[0]
        self.malrewards[timestep,1,1] = detection_reward[1]
        
    def update_malQ(self, alpha, W, timestep):
        """
        malQ : array
            Q-value for each action, where array column is 'cyber' (0) and 'misinformation' (1)
            and row is provider id
        """
        if len(self.malagents) > 0 and W < timestep:        
            if self.attack_decisions[timestep-1]==1: # Attack decision is 'attack'
                target = self.targets[np.where(self.attack_decisions==1)[0][0]]
                if self.attack_methods[timestep-1] == 0: # Attack method is cyber attack
                    self.malQ[target,0] = self.malQ[target,0]+\
                        alpha * (np.sum(self.malrewards[timestep,:,0]) - self.malQ[target,0])
                elif self.attack_methods[timestep-1] == 1: # Attack method is misinformation
                    self.malQ[target,1] = self.malQ[target,1]+\
                        alpha * (np.sum(self.malrewards[timestep,:,1]) - self.malQ[target,1])
                else: # Both attack methods used simultaneously
                    self.malQ[target,0] = self.malQ[target,0]+\
                        alpha * (np.sum(self.malrewards[timestep,:,0]) - self.malQ[target,0])
                    self.malQ[target,1] = self.malQ[target,1]+\
                        alpha * (np.sum(self.malrewards[timestep,:,1]) - self.malQ[target,1])
            else:
                provider = self.targets[timestep-1]
                if self.attack_methods[timestep-1] == 0: # Observe cyber attack
                    self.malQ[provider,2] = self.malQ[provider,2]+\
                        alpha * (np.sum(self.malrewards[timestep,:,0]) - self.malQ[provider,2])
                else: # Observe misinformation
                    self.malQ[provider,3] = self.malQ[provider,3]+\
                        alpha * (np.sum(self.malrewards[timestep,:,1]) - self.malQ[provider,3])

    def spread_misinformation(self, neighbour, opinion, delta):
        """
        feedback : int
            Feedback value for spreading misinformation, where 0 is no feedback and 1 is agreement
            and -1 is disagreement
        """
        # Check if target has been selected and opinion about target's behaviour is asked
        indices = np.where(self.attack_decisions==1)[0]
        if indices.size > 0 and opinion[0] == self.targets[indices[0]]:
                if opinion[1] == 1: # If positive opinion about target is asked
                    feedback = -1 # Disagree with target's positive behaviour
                else:
                    feedback = 1 # Agree with target's negative behaviour
        elif self.network.nodes[neighbour]["n(a)"][opinion[0]] >= 1: # Check if agent has information; speak truth
            if self.network.nodes[neighbour]["n(r)"][opinion[0]]/self.network.nodes[neighbour]["n(a)"][opinion[0]] >= delta:
                satisfaction = 1 # Satisfied with provider's behaviour
            else:
                satisfaction = -1 # Unsatisfied with provider's behaviour
            if satisfaction == opinion[1]: # Check if can agree (satisfaction matches with opinion)
                feedback = 1 # Agree
            else:
                feedback = -1 # Disagree
        else: 
            feedback = 0 # Give no information
        return feedback
    
    def generate_attack_edges(self, timestep):
        """
        Generate attack edges for current timestep based on active attacks
        Returns list of attack edges with layer="attack" and appropriate phase
        """
        attack_edges = []
        
        # Check if attack is active
        if (timestep < len(self.attack_decisions) and 
            self.attack_decisions[timestep] == 1 and
            timestep < len(self.targets) and 
            self.targets[timestep] != -1 and
            timestep < len(self.attack_methods) and
            self.attack_methods[timestep] != -1):
            
            target = self.targets[timestep]
            attack_method = self.attack_methods[timestep]
            
            # Map attack method to phase
            phase_map = {0: "CYBER", 1: "MISINFO", 2: "COMBO"}
            phase = phase_map.get(attack_method, "NONE")
            
            # Generate attack edges from each malicious agent to target provider
            for mal_agent in self.malagents:
                edge = {
                    "id": f"attack-{mal_agent}-P-{target}",
                    "source": str(mal_agent),
                    "target": f"P-{target}",
                    "layer": "attack",
                    "phase": phase,
                    "dir": "→",
                    "type": "attack"  # For backward compatibility
                }
                attack_edges.append(edge)
        
        return attack_edges
    
    def get_active_attack_info(self, timestep):
        """
        Get information about currently active attacks
        Returns dict with attack status, target, and method
        """
        attack_info = {
            "active": False,
            "target": -1,
            "method": -1,
            "phase": "NONE"
        }
        
        if (timestep < len(self.attack_decisions) and 
            self.attack_decisions[timestep] == 1 and
            timestep < len(self.targets) and 
            self.targets[timestep] != -1 and
            timestep < len(self.attack_methods) and
            self.attack_methods[timestep] != -1):
            
            phase_map = {0: "CYBER", 1: "MISINFO", 2: "COMBO"}
            
            attack_info["active"] = True
            attack_info["target"] = self.targets[timestep]
            attack_info["method"] = self.attack_methods[timestep]
            attack_info["phase"] = phase_map.get(self.attack_methods[timestep], "NONE")
        
        return attack_info
