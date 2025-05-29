#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) [2024] [Kart Padur]

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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timer import Timer
from experiment import experiment
from setup_values import timestep, nTimesteps, nExperiments, nProviders, nAgents, nMalAgents, W, save_fig, output_dir,\
    alpha, delta, epsilon, eta, theta, kappa, Lambda, lambda_, xi,\
    rho, tau, Upsilon, upsilon_, experiment_id
run = 1

for i in range(run):
    tick = Timer()
    tick.start()
    # Collect relevant data
    data_df = pd.DataFrame(columns = ['regret','provider', 'avguse', 'avgreward', 'posopinion', 'negopinion', 
        'avgusebefore', 'avguseduring', 'avguseafter', 'avgrewbefore', 'avgrewduring', 'avgrewafter', 
        'posopinionbefore', 'posopinionduring', 'posopinionafter', 'negopinionbefore', 'negopinionduring', 'negopinionafter',
        'target_id', 'attack_beginning', 'misinfo_beginning', 'misinfo_length', 'cyber_beginning', 'cyber_length',
        'combined_beginning','combined_length','impactusage', 'impactposop', 'impactnegop','attackreward', 'detectionreward', 'malreward',
        'alpha','delta', 'epsilon', 'eta', 'theta', 'kappa', 'Lambda', 'lambda', 'xi', 'rho', 'tau', 'Upsilon', 'upsilon'])

    # Run experiment
    for i in range(nExperiments):
        print("[Experiment {}/{}]".format(i + 1, nExperiments))
        actions, rewards, opinionvalues, regrets, attack_decisions, targets, attack_methods, \
        attack_rewards, detection_rewards = experiment(timestep, nTimesteps)
        # Calculate average regret value
        avgregret = np.mean(regrets[W+1:])/nAgents
        # Collect information if an attack happened
        if nMalAgents > 0 and np.any(attack_decisions == 1):
            attack_timesteps = np.array(np.where(attack_decisions==1))[0]
            target_provider =  targets[attack_timesteps[0]] # attack target
            misinfo_campaign1 = np.where((attack_methods == 1) & (attack_decisions==1))[0]
            misinfo_campaign2 = np.where((attack_methods == 2) & (attack_decisions==1))[0]
            misinfo_campaign = [*misinfo_campaign1, *misinfo_campaign2]
            misinfo_campaign.sort()
            misinfo_length = len(misinfo_campaign)
            if len(misinfo_campaign)>=1:
                misinfo_beginning = min(misinfo_campaign)+1
            else:
                misinfo_beginning = ''
            cyber_campaign1 = np.array(np.where((attack_methods == 0) & (attack_decisions==1)))[0]
            cyber_campaign2 = np.array(np.where((attack_methods == 2) & (attack_decisions==1)))[0]
            cyber_campaign = [*cyber_campaign1, *cyber_campaign2]
            cyber_campaign.sort()
            cyber_length = len(cyber_campaign)
            if len(cyber_campaign)>=1:
                cyber_beginning = min(cyber_campaign)+1
            else:
                cyber_beginning = ''
            combined_campaign = np.array(np.where((attack_methods == 2) & (attack_decisions==1)))[0]
            combined_length = len(combined_campaign)
            if combined_length == 0:
                combined_beginning = ''
            else:
                combined_beginning = min(combined_campaign)+1
            if nTimesteps-1 == attack_timesteps[-1]:
                reward_timesteps = np.append(attack_timesteps[1:],  attack_timesteps[-1])
            else:
                reward_timesteps = np.append(attack_timesteps[1:],  attack_timesteps[-1]+1)
            attackreward = np.sum(attack_rewards[reward_timesteps])
            detectionreward = np.sum(detection_rewards[reward_timesteps])
            malreward = attackreward + detectionreward 
            
            print("Target: ", target_provider+1, ", attack timesteps: " , attack_timesteps\
                , ", length of attack campaign: ", len(attack_timesteps)\
                , ", misinfo campaign: ", misinfo_campaign, ", cyber campaign: ", cyber_campaign)
        for provider in range(nProviders):
            # Calculate average use for each provider
            avguse = np.mean(actions[W+1:,provider])/nAgents
            # Calculate average reward value for each provider
            avgreward = np.sum(rewards[W+1:,provider])/np.sum(actions[W+1:,provider])
            # Calculate average positive opinion value for each provider
            posopinion = np.mean(opinionvalues[W+1:,provider+3])
            # Calculate average negative opinion value for each provider
            negopinion =  np.mean(opinionvalues[W+1:,provider])
            # malagents
            if nMalAgents > 0 and np.any(attack_decisions == 1):
                avgusebefore = np.mean(actions[attack_timesteps[0]-len(attack_timesteps):attack_timesteps[0], provider])/nAgents
                avguseduring = np.mean(actions[attack_timesteps, provider])/nAgents
                avguseafter = np.mean(actions[attack_timesteps[-1]:attack_timesteps[-1]+len(attack_timesteps), provider])/nAgents
                avgrewbefore = np.sum(rewards[attack_timesteps[0]-len(attack_timesteps):attack_timesteps[0], provider])/\
                    np.sum(actions[W+1:attack_timesteps[0],provider])
                avgrewduring = np.sum(rewards[attack_timesteps, provider])/np.sum(actions[attack_timesteps,provider])
                avgrewafter = np.sum(rewards[attack_timesteps[-1]:attack_timesteps[-1]+len(attack_timesteps), provider])/\
                    np.sum(actions[attack_timesteps[-1]:(nTimesteps-1),provider])
                posopinionbefore = np.mean(opinionvalues[attack_timesteps[0]-len(attack_timesteps):attack_timesteps[0], provider+3])
                posopinionduring =  np.mean(opinionvalues[attack_timesteps, provider+3])
                posopinionafter =  np.mean(opinionvalues[attack_timesteps[-1]:attack_timesteps[-1]+len(attack_timesteps), provider+3])
                negopinionbefore =  np.mean(opinionvalues[attack_timesteps[0]-len(attack_timesteps):attack_timesteps[0], provider])
                negopinionduring =  np.mean(opinionvalues[attack_timesteps, provider])
                negopinionafter =  np.mean(opinionvalues[attack_timesteps[-1]:attack_timesteps[-1]+len(attack_timesteps), provider])
                # impact on usage and opinions
                impactusage = avguse - avguseduring
                impactposop = posopinion - posopinionduring
                impactnegop = abs(negopinion) - abs(negopinionduring)
                data_df = pd.concat([data_df, pd.DataFrame.from_records([{'regret':avgregret, 'provider':int(provider+1), 'avguse':avguse, 'avgreward':avgreward,
                    'posopinion':posopinion, 'negopinion':negopinion, 'avgusebefore':avgusebefore, 'avguseduring':avguseduring, 
                    'avguseafter':avguseafter, 'avgrewbefore':avgrewbefore, 'avgrewduring':avgrewduring, 'avgrewafter':avgrewafter,
                    'posopinionbefore':posopinionbefore, 'posopinionduring':posopinionduring, 'posopinionafter':posopinionafter, 
                    'negopinionbefore':negopinionbefore, 'negopinionduring':negopinionduring, 'negopinionafter':negopinionafter, 
                    'target_id':target_provider+1, 'attack_beginning':attack_timesteps[0]+1,
                    'misinfo_beginning':misinfo_beginning, 'misinfo_length':misinfo_length, 'cyber_beginning':cyber_beginning, 
                    'cyber_length':cyber_length, 'combined_beginning': combined_beginning, 'combined_length': combined_length,
                    'impactusage':impactusage, 'impactposop':impactposop, 'impactnegop': impactnegop,
                    'attackreward':attackreward, 'detectionreward':detectionreward, 'malreward':malreward,
                    'alpha': alpha, 'delta': delta, 'epsilon': epsilon, 'eta': eta, 'theta': theta, 'kappa': kappa,
                    'Lambda': Lambda[0], 'lambda': lambda_[0], 'xi': xi, 'rho': rho, 'tau': tau, 
                    'Upsilon': Upsilon[0], 'upsilon': upsilon_[0]}])])
            else:
                data_df = pd.concat([data_df, pd.DataFrame.from_records([{'regret':avgregret, 'provider':int(provider+1), 'avguse':avguse, 'avgreward':avgreward,
                    'posopinion':posopinion, 'negopinion':negopinion, 'avgusebefore':'', 'avguseduring':'', 'avguseafter':'', 
                    'avgrewbefore':'', 'avgrewduring':'', 'avgrewafter':'',
                    'posopinionbefore':'', 'posopinionduring':'', 'posopinionafter':'', 
                    'negopinionbefore':'', 'negopinionduring':'', 'negopinionafter':'', 
                    'target_id':'', 'attack_beginning':'',
                    'misinfo_beginning':'', 'misinfo_length':'', 'cyber_beyginning':'', 
                    'cyber_length':'', 'combined_beginning': '', 'combined_length': '',
                    'impactusage':'', 'impactposop':'' , 'impactnegop': '',
                    'attackreward':'', 'detectionreward':'', 'malreward':'',
                    'alpha': alpha, 'delta': delta, 'epsilon': epsilon, 'eta': eta, 'theta': theta, 
                    'kappa': kappa, 'Lambda': Lambda[0], 'lambda': lambda_[0], 'xi': xi, 'rho': rho, 'tau': tau, 
                    'Upsilon': Upsilon[0], 'upsilon': upsilon_[0]}])])
    
    tick.stop()

    # Write data locally
    output_dir_data = os.path.join(os.getcwd(), "output_data")
    with open(output_dir_data+str(f'/experiment{experiment_id}_df.csv'), 'a', encoding='UTF8', newline='') as f:
        data_df.to_csv(f, header=False)

    # Visualise data
    plt.rcParams["figure.figsize"] = (20,4)

    # Plot 1: Visualise actions per provider over time
    A_over_time = (actions / nExperiments)*100/nAgents
    for provider in range(nProviders):
        timesteps = list(np.array(range(len(A_over_time[:,provider])))+1)
        plt.plot(timesteps, A_over_time[:,provider], "-", label="Provider {}".format((provider+1)))
    plt.xlabel("Time steps")
    plt.ylabel("Average provider usage (%)")
    plt.xlim([1, nTimesteps])
    plt.ylim([1, 100])
    plt.grid()
    ax = plt.gca()
    leg = ax.legend(loc='center left', shadow=True, bbox_to_anchor =(1, 0.5))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)
    if save_fig:
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        plt.rcParams["figure.figsize"] = (19,4)
        plt.savefig(os.path.join(output_dir, "actions.pdf"), dpi=500, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    
    # Plot 2: Visualise average collected reward per provide over time
    aveR_A = rewards / actions
    for provider in range(nProviders):
        timesteps = list(np.array(range(len(aveR_A[:,provider])))+1)
        plt.plot(timesteps, aveR_A[:,provider], ".", label="Provider {}".format((provider+1)))
    plt.xlabel("Time steps")
    plt.ylabel("Average reward per service provider")
    plt.xlim([1, nTimesteps])
    plt.ylim([-0.01, 1.01])
    plt.grid()  
    ax = plt.gca()
    leg = ax.legend(loc='center left', shadow=True, bbox_to_anchor =(1, 0.5))
    if save_fig:
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        plt.rcParams["figure.figsize"] = (19,4)
        plt.savefig(os.path.join(output_dir, "rewards.pdf"), dpi=500, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    
    # Plot 3: Visualise opinion dynamics over time
    for opinion in range(nProviders*2):
        timesteps = list(np.array(range(len(opinionvalues[:,opinion])))+1)
        if opinion in [0]:
            plt.plot(timesteps, opinionvalues[:,opinion], "--", color = 'tab:blue', label = "Provider  {} -".format(opinion+1)) # 1
        elif opinion in [1]:
            plt.plot(timesteps, opinionvalues[:,opinion], "--", color = 'tab:orange', label = "Provider  {} - ".format(opinion+1)) # 2
        elif opinion in [2]:
            plt.plot(timesteps, opinionvalues[:,opinion], "--", color = 'tab:green', label = "Provider  {} - ".format(opinion+1)) # 3
        elif opinion in [3]:
            plt.plot(timesteps, opinionvalues[:,opinion], "-", color = 'tab:blue', label = "Provider  {} + ".format(opinion-2)) # 1
        elif opinion in [4]:
            plt.plot(timesteps, opinionvalues[:,opinion], "-", color = 'tab:orange', label = "Provider  {} + ".format(opinion-2)) # 2
        else:
            plt.plot(timesteps, opinionvalues[:,opinion], "-", color = 'tab:green', label = "Provider  {} + ".format(opinion-2)) # 3
    plt.xlabel("Time steps")
    plt.ylabel("Average evalution of opinion value per service provider")
    plt.xlim([1, nTimesteps])
    plt.grid()
    ax = plt.gca()
    leg = ax.legend(loc='center left', shadow=True, bbox_to_anchor =(1, 0.5))
    if save_fig:
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        plt.rcParams["figure.figsize"] = (19,4)
        plt.savefig(os.path.join(output_dir, "opinionvalues.pdf"), dpi=500, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    
    # Plot 4: Visualise malicious agents' reward values over time
    timesteps = list(np.array(range(len(attack_rewards)))+1)
    plt.plot(timesteps, attack_rewards, ":", color = 'tab:purple', label="Attack rewards")
    plt.plot(timesteps, detection_rewards, ":", color = 'tab:cyan', label="Detection rewards")
    plt.xlabel("Time steps")
    plt.ylabel("Malicious users' rewards")
    plt.xlim([1, nTimesteps])
    plt.grid()
    ax = plt.gca()
    leg = ax.legend(loc='center left', shadow=True, bbox_to_anchor =(1, 0.5))
    if save_fig:
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        plt.rcParams["figure.figsize"] = (19,4)
        plt.savefig(os.path.join(output_dir, "malrewards.pdf"), dpi=500, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    
    i+=1