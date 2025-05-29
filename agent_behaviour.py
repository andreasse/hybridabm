#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
np.random.seed(0)

class AgentsBehaviour:
    """Agents' behaviour"""

    def __init__(self, network, providers):
        """Agents use an epsilon-greedy action selection method and Q-learning for action-value estimates.
        Agents use an epsilon-greedy opinion selection method and stateless Q-learning for opinin-value estimates."""
        self.network = network
        self.providers = providers
    
    def select_action(self, agent, epsilon, tau, timestep):
        """
        action : int
            Provider whose service is taken in this time step; depends only on user's
            experience 
        """
        # Select action using epsilon-greedy policy
        if timestep == 1 or np.random.uniform(0, 1) < epsilon: # Explore actions
            action = np.random.choice(self.providers)
        else: # Exploit learned values
            Q_values = np.zeros(len(self.providers), dtype = float)
            # Combine action values with opinion values to determine the next action
            for provider in range(len(self.providers)):
                value = tau * self.network.nodes[agent]["Q(a)"][provider]+\
                    (1-tau) * (self.network.nodes[agent]["Phi(o)"][provider+len(self.providers)]\
                        -self.network.nodes[agent]["Phi(o)"][provider])
                Q_values[provider] = value
            # Choose action with the highest value
            action = np.random.choice(np.flatnonzero(Q_values == Q_values.max()))
        # Update the actions vector
        self.network.nodes[agent]["a"][timestep] = action
        # Update the count of actions' vector
        self.network.nodes[agent]["n(a)"][action] += 1
        return action
    
    def receive_reward(self, agent, action, endpoint, timestep):
        """
        reward : binary int
            End-point service level of the selected provider
        """
        # Collect reward for action
        reward = endpoint[agent,timestep,action]
        # Add reward value to rewards history
        self.network.nodes[agent]["r(a)"][timestep] = reward
        # Update the vector of positive rewards
        self.network.nodes[agent]["n(r)"][action] += reward
        return reward

    def update_Q(self, agent, action, reward, alpha):
        """
        An agent uses off-policy TD control algorithm: Q-learning
        """
        # Updated action value estimates using stateless Q-learning
        # Q(a) <- Q(a) + alpha * (r - Q(a))
        self.network.nodes[agent]["Q(a)"][action] = self.network.nodes[agent]["Q(a)"][action]\
            + alpha * (reward - self.network.nodes[agent]["Q(a)"][action])
        return self.network.nodes[agent]["Q(a)"]

    def express_opinion(self, agent, epsilon, timestep):
        """
        opinion : tuple (service provider id, value)
            Opinion on one service provider which is expressed to a randomly chosen
            neighbour. It is chosen using epsilon-greedy policy.
        """
        # Select action using epsilon-greedy policy
        if timestep == 1 or np.random.uniform(0, 1) < epsilon: # Explore opinions
            provider = np.random.choice(self.providers) # choose provider
            value = np.random.choice([-1,1]) # express opinion
            if value == 1: index = provider + len(self.providers) # determine opinion index (0-5)
            else: index = provider
        else: # Exploit learned values
            index = np.where(self.network.nodes[agent]["Phi(o)"] \
                == self.network.nodes[agent]["Phi(o)"].max())[0][0]
            opinionvalues = np.repeat(np.array([-1,1]),len(self.providers))
            value = opinionvalues[index] # -1 or 1
            if index > len(self.providers)-1: provider = index - len(self.providers) # provider id
            else: provider = index
        opinion = (provider, value, index)
        return opinion
    
    def find_neighbour(self, agent, timestep):
        """
        neighbour : int (neighbour's id)
            A neighbour who is available for changing information
        """
        # Randomly choose a neighbour from whom to ask information
        neighbour = random.choice(self.network.nodes[agent]["C"])
        # Add neighbour to communication partners' list
        self.network.nodes[agent]["zeta"][timestep] = neighbour
        return neighbour
    
    def ask_info(self, neighbour, malagents, mal_behaviour, opinion, delta,
                 attack_decision, attack_method):
        """
        feedback : int (1,0,-1)
            Feedback describes whether the selected neighbour agrees with agent's
            opinion, has no previous experience with it, or disagrees.
        """
        # Express opinion to the selected neighbour to get their feedback
        if neighbour in malagents and attack_decision==1 and (attack_method==1 or attack_method==2):
            return mal_behaviour.spread_misinformation(neighbour, opinion, delta)
        else:
            return self.provide_feedback(neighbour, opinion, delta)

    def provide_feedback(self, neighbour, opinion, delta):
        """
        feedback : int (1,0,-1)
            Feedback describes whether the selected neighbour agrees with agent's
            opinion, has no previous experience with it, or disagrees. Evaluation
            of current opinion on provider is evaluated using a threshold value d.
        """
        # Give feedback for a neighbour who asked information
        if self.network.nodes[neighbour]["n(a)"][opinion[0]] >= 1: # Check if agent has information on provider
            # Check if service provider satisfaction level exeeds threshold
            if self.network.nodes[neighbour]["n(r)"][opinion[0]]/self.network.nodes[neighbour]["n(a)"][opinion[0]] >= delta:
                satisfaction = 1 # Satisfied as current experience value exceeds threshold value
            else:
                satisfaction = -1 # Unsatisfied
            if satisfaction == opinion[1]: # Check if satisfaction matches with opinion
                feedback = 1 # Approve 
            else:
                feedback = -1 # Disapprove
        else:
            feedback = 0 # Give no feedback
        return feedback

    def update_opinion_value(self, agent, opinion, feedback, alpha):
        """
        An agent uses stateless Q-learning to form an internal evaluation of opinion
        """
        # Updated opinion value estimates using Q-learning, when received negative/positive feedback
        # Q(o) <- Q(o) + alpha * (r - Q(o))
        if feedback != 0:
            self.network.nodes[agent]["Phi(o)"][opinion[2]] =\
                    self.network.nodes[agent]["Phi(o)"][opinion[2]] + alpha *\
                    (feedback - self.network.nodes[agent]["Phi(o)"][opinion[2]])
        return self.network.nodes[agent]["Phi(o)"]
