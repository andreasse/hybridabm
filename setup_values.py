#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# scenarios - change these to change the experiment
cyberattack = False
misinformation = False
coordinated_attack = True

# Deteremine experiment id
if cyberattack:
    experiment_id = 1
elif misinformation:
    experiment_id = 2
elif coordinated_attack:
    experiment_id = 3
else:
    experiment_id = 0 # no attack

timestep = 1
nExperiments = 1
#nTimesteps = 10000
nTimesteps = 3000
#W = 2000
W = 10
nAgents = 50
if cyberattack or misinformation or coordinated_attack:
    nMalAgents = int(0.05*nAgents)
else: # no malicious users are introduced if no attack is selected
    nMalAgents = 0
regagents = nAgents - nMalAgents
nProviders = 3

# Parameters (NOTE: some parameters have updated notation added in parentheses)
alpha = 0.05 # learning rate
epsilon = 0.10 # exploration parameter
theta = 0.80 # cyberattack detection probability (paper: zeta)
eta = 0.10 # misinfo detection probability
kappa = 6 # number of neighbours
rho = 0.01 # rewiring probability
delta = 0.80 # delta - satisfaction threshold (paper: tau)
tau = 0.80 # direct experience weight (paper: omega)
xi = 100 # positive initialisation of malicious agents
Lambda = [0.03,0.03,0.03] # endpoint 1 to 0 (paper: sigma)
lambda_ = [0.25,0.25,0.25] # endpoint 0 to 1 (paper: theta)
Upsilon = [0.01,0.01,0.01] # central 1 to 0 (latest: upsilon)
upsilon_ = [0.30,0.30,0.30] # central 0 to 1 (paper: psi)

# Save figures
save_fig = True
output_dir = os.path.join(os.getcwd(), "output_fig")
