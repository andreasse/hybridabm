#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np

class ServiceProvidersBehaviour:
    """Service providers' behaviour"""

    def __init__(self, providers, agents, Upsilon, upsilon_, Lambda, lambda_, nTimesteps):
        """A two-state Markov model is used to model the central service states and
        end-point service levels of each provider. Each provider has different transition
        probabilities for central service states and end-point service levels"""
        self.providers = providers
        self.agents = agents
        self.Upsilon = Upsilon
        self.upsilon_ = upsilon_
        self.Lambda = Lambda
        self.lambda_ = lambda_
        self.nTimesteps = nTimesteps
        self.center = np.full((self.nTimesteps+1, len(self.providers)), fill_value = 1, dtype = int)
        self.endpoint = np.full((len(self.agents), self.nTimesteps+1, len(self.providers)),
                                fill_value = 1, dtype = int)
        self.serviceStates = [1,0]
        self.serviceLevels = [1,0]
        self.center_vector = np.zeros((self.nTimesteps, len(self.providers), 2), dtype = float)
        self.endpoint_vector = np.zeros((self.nTimesteps, len(self.providers), 2), dtype = float)
        
        # NEW: Sentiment tracking for each provider over time
        self.sentiment = np.zeros((self.nTimesteps+1, len(self.providers)), dtype=float)

    def show_init_central_state(self):
        """
        center : array
            Central service states.
        """
        return self.center

    def central_state(self, provider, target, attack_method, attack_decision, timestep):
        """
        center : array
            Central service states modelled with two-state Markov model.
        """
        if timestep == 1:
                cState = 1
        elif attack_decision==1 and target == provider and (attack_method == 0 or attack_method == 2): # 0 - cyberattack, 2 - cyber & misinfo
            # Note: should be probabilistic - there is some probability that the central state will
            # 'survive' an attack and not go unavailable.
            cState = 0
        elif timestep == 1:
            cState = 1
        elif self.center[timestep-1,provider] == 1:
            cState = random.choices(self.serviceStates,\
                            weights = ((1-self.Upsilon[provider]), self.Upsilon[provider]), k = 1)[0]
        else:
            cState = random.choices(self.serviceStates,\
                            weights = (self.upsilon_[provider], (1-self.upsilon_[provider])), k = 1)[0]
        self.center[timestep,provider]=cState
        return self.center

    def endpoint_level(self, provider, center, timestep):
        """
        endpoint : array
            Endpoint service states modelled with two-state Markov model.
        """
        for agent in self.agents:
            if timestep == 1:
                    eLevel = 1
            elif center[timestep,provider]==1 and self.endpoint[agent,timestep-1,provider] == 1:
                eLevel = random.choices(self.serviceLevels,\
                            weights = ((1-self.Lambda[provider]), self.Lambda[provider]), k = 1)[0]
            elif center[timestep,provider]==1 and self.endpoint[agent,timestep-1,provider] == 0:
                eLevel = random.choices(self.serviceLevels,\
                            weights = (self.lambda_[provider], (1-self.lambda_[provider])), k = 1)[0]
            else:
                eLevel = 0
            self.endpoint[agent,timestep,provider] = eLevel
        return self.endpoint
    
    def calculate_sentiment(self, network, timestep):
        """
        Calculate sentiment for each provider based on agent opinions
        Sentiment is the mean of positive opinions minus negative opinions
        
        network: NetworkX graph containing agent opinion data
        timestep: current timestep
        """
        for provider in range(len(self.providers)):
            positive_opinions = []
            negative_opinions = []
            
            # Collect opinions from all agents
            for agent in self.agents:
                if "Phi(o)" in network.nodes[agent]:
                    # Positive opinion index: provider + len(providers)  
                    # Negative opinion index: provider
                    pos_idx = provider + len(self.providers)
                    neg_idx = provider
                    
                    if pos_idx < len(network.nodes[agent]["Phi(o)"]):
                        positive_opinions.append(network.nodes[agent]["Phi(o)"][pos_idx])
                    
                    if neg_idx < len(network.nodes[agent]["Phi(o)"]):
                        negative_opinions.append(network.nodes[agent]["Phi(o)"][neg_idx])
            
            # Calculate sentiment as difference between positive and negative opinions
            avg_positive = np.mean(positive_opinions) if positive_opinions else 0
            avg_negative = np.mean(negative_opinions) if negative_opinions else 0
            
            # Normalize sentiment to [-1, 1] range
            sentiment = np.tanh(avg_positive - avg_negative)
            self.sentiment[timestep, provider] = sentiment
        
        return self.sentiment[timestep]
    
    def get_provider_stats(self, timestep):
        """
        Get provider statistics for export to frontend
        Returns dict with sentiment and down status for each provider
        """
        provider_stats = {}
        
        for provider in range(len(self.providers)):
            provider_id = f"P-{provider}"
            provider_stats[provider_id] = {
                "sentiment": float(self.sentiment[timestep, provider]),
                "down": bool(self.center[timestep, provider] == 0)  # zd == 0 means down
            }
        
        return provider_stats
    