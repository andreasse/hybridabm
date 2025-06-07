# hybridabm

# Using Agent-Based Modelling and Reinforcement Learning to Study Hybrid Threats
## Purpose and patterns
Hybrid attacks coordinate the exploitation of vulnerabilities across domains to undermine trust in authorities and cause social unrest. Whilst such attacks have primarily been seen in active conflict zones, there is growing concern about the potential harm that can be caused by hybrid attacks more generally and a desire to discover how better to identify and react to them. In addressing such threats, it is important to be able to identify and understand an adversary's behaviour. For this purpose, we present a novel agent-based model in which, for the first time, agents use reinforcement learning to inform their decisions. We aim to understand how adversaries can develop their attack strategy based only on their experience with and observations of the dynamics of the surrounding system. We evaluate the impact of such attacks on the behaviours and opinions of the target audience. We demonstrate the face validity of this approach and argue that its generality and adaptability render it an important tool in formulating holistic responses to hybrid threats, including proactive vulnerability identification, which does not necessarily emerge by considering the multiple threat vectors independently.

## Entities, state variables, and scales
The following entities are included in the model: 
 - **Service providers** that represent *Cyber-Physical Systems* with central and end-point states,
 - **Regular agents** are part of the *Social System* and can interact with their social connections in the social network and request service from the service providers,
 - **Malicious agents** are part of the *Social System* and can coordinate cyber and disinformation attacks against a service provider,
 - **Environment** represents a *Cyber-Physical-Social System* (CPSS). Next to physical components and supporting digital elements, a CPSS considers humans and social dynamics as an integral part of the whole system.

Service providers belong to a set of service providers. They have central and end-point states that can be either $1$ or $0$ indicating availability. Regular agents belong to a set of regular agents that can be identified with an ID, and malicious agents belong to a set of malicious agents that can also be identified with an ID.

## Process overview and scheduling
1) The timer starts with running the execution of the experiment (```python run.py```)
2) Environment, regular agents, service providers, and malicious agents are created

*For each time step*:

3) The environment displays its state
4) Environment calculates rewards for malicious agents (from the second time step)
5) Malicious agents update their policy (from the second time step)
6) Malicious agents take action
7) Service providers change their state (depending on transition probabilities and on an ongoing attack)
8) Regular agents take action and express opinion
9) Environment calculates rewards for regular agents' actions
10) Regular agents update their policy for choosing actions based on the received rewards
11) Regular agents express their opinions to their neighbour
12) Environment calculates rewards for regular agents' opinions
13) Regular agents update their policy for choosing opinions to express based on their received rewards
14) Data is collected for visualisation
15) After the terminal time step, the experiment is completed, and results are visualised

This process follows a general reinforcement learning framework, where agents and the environment interact sequentially to facilitate learning. Malicious agents take action first because their behaviour directly impacts the service provider's state, shaping the subsequent experience of regular agents with the service provider. This version emphasises the cause-and-effect relationship between malicious agents' actions and how they influence regular agents' interactions.

## Design concepts
### Basic principles
This model combines *agent-based modelling* (ABM) and *reinforcement learning* (RL) to simulate the dynamics of hybrid threats—coordinated cyberattacks and misinformation—within a system of interacting agents. ABM captures the interactions between malicious and regular agents in a shared environment, drawing from theories of complex adaptive systems commonly used in cybersecurity and misinformation studies to explore emergent system-level behaviours.

Malicious and regular agents use RL to learn and adapt their actions over time. Malicious agents optimise their attacks based on the feedback they receive from the environment, aiming to disrupt service providers and influence the experiences of regular agents. Regular agents, in turn, use RL to adjust their responses to changes in the environment, such as disruptions caused by cyberattacks or exposure to misinformation. This approach reflects adaptive behaviour, where agents learn and evolve based on changes in the environment.

### Emergence
In this model, the system-level behaviours primarily emerge from the adaptive decisions and interactions of malicious and regular agents, who learn through RL in response to environmental feedback. 

Key emergent results include the collective impact of hybrid threats on the system and how regular agents modify their behaviour based on disruptions caused by malicious agents. For example, small changes in malicious agents' strategies or the environment (such as increased unavailability of service providers) may lead to disproportionately large effects on regular agents’ experiences, highlighting the nonlinear dynamics of the system. These results emerge from the interaction between malicious agents’ adaptive attacks, regular agents’ evolving behaviour, and the state of the service providers, creating a system where outcomes are sensitive to the behaviour of all agents involved. 

On the other hand, some aspects of the model are imposed by predefined rules or mechanisms. For example, the structure of the social network is fixed and not subject to change. Similarly, how the environment provides feedback to the agents (e.g., service disruptions or exposure to misinformation) follows specific rules and does not evolve over time. These imposed features are relatively predictable, ensuring that the framework within which agents operate remains stable while the agent-environment interactions drive emergent outcomes.

### Adaptation
In this model, malicious and regular agents exhibit adaptive behaviours through RL. Each agent makes decisions based on feedback from the environment, adapting their actions to achieve their respective goals. 

Malicious agents choose from a set of attack strategies, including cyberattacks and misinformation campaigns, based on their impact on regular agents. Their decision-making is driven by an internal objective: maximising the number of service provider customers who experience attacks by directly experiencing service unavailability or through exposure to negative information on the social network, both of which may cause customers to change their provider. 

On the other hand, regular agents adapt their responses to the state of the environment, which has been influenced by the malicious agents’ actions. Given ongoing disruptions or misinformation, they decide how to interact with the service providers,. These agents are also objective-seeking, optimising their behaviour to maximise the rewards they receive. Internal factors, such as learned preferences or past experiences, and environmental factors, such as the state of the service providers, guide their decisions.

### Objectives
In this model, malicious and regular agents use direct objective-seeking to guide their adaptive behaviours. Each agent evaluates actions based on a specific objective measure representing their success.

For malicious agents, the objective is to maximise the number of service provider customers who experience attacks by directly encountering service unavailability or through exposure to negative information on the social network. This objective measure is driven by variables such as the current state of the service provider’s availability, the spread of misinformation, and customer reactions to these factors. The malicious agents select actions that lead to the highest customer disruption or defection to another provider. The calculation of this objective involves estimating how different attack strategies—whether cyberattacks or misinformation campaigns—will influence customer behaviour. The rationale is that the more customers they affect, the greater the success of their coordinated hybrid threats.

For regular agents, the objective measure remains the utility they gain from interacting with service providers and expressing opinions. They aim to maximise their satisfaction by selecting providers that offer reliable service and expressing opinions that others approve of. Their objective is calculated by evaluating the expected benefit of staying with or switching to a different provider based on service reliability and trustworthiness.

### Learning
In the model, malicious and regular agents use RL to adapt their decision-making over time based on experience. As agents interact with the environment and receive feedback, they update their strategies to achieve their objectives better. Learning is modelled through an RL framework, where agents adjust their actions based on the rewards or penalties they receive, making the decision-making process dynamic and adaptive. This representation is grounded in existing RL theory, and it is included to simulate how both malicious and regular agents evolve their strategies to optimise their respective goals—whether it is maximising harm or reward.

### Prediction
The model incorporates implicit prediction in the agents' decision-making. Malicious and regular agents make decisions based on assumptions about how their actions will affect future outcomes. Malicious agents implicitly predict that an attack will disrupt service providers and influence customers to switch providers. In contrast, regular agents predict that staying with or switching to a service provider will impact their cumulative reward. These predictions are not modelled explicitly but are assumed to guide the agents' adaptive behaviours in a way that mimics real-world decision-making under uncertainty. The rationale for this approach is to focus on useful agent behaviour without overcomplicating the prediction process.

### Sensing
Agents are assumed to sense critical variables in their environment. Malicious agents sense the current state of service providers (availability, vulnerability) and the spread of misinformation, while regular agents sense service quality and information on social networks. Both agent types sense their own internal states, such as past rewards or penalties from their decisions. The model assumes that agents accurately sense these variables without uncertainty. This assumption simplifies the agents' decision-making while allowing the model to capture their interactions with a dynamic environment.

### Interaction
Interactions in the model are represented as a mix of direct and mediated interactions. Malicious agents interact directly with service providers through cyberattacks and spread misinformation to regular agents. In contrast, regular agents experience these interactions indirectly as they are affected by service disruptions or exposed to negative information. Mediated interactions occur through the social network, where misinformation spreads and influences customer behaviour. The range of these interactions is determined by the structure of the network and the reach of malicious actions to model the complex interplay between attacks and customer decisions.

### Stochasticity
Stochastic processes are used in the model to introduce variability in both the spread of misinformation and customer reactions to service provider disruptions. This randomness allows the model to simulate more realistic, unpredictable outcomes, reflecting that not all customers respond to information or service outages similarly. Stochasticity is also applied when determining initial conditions, such as customer preferences and initial service provider performance, to ensure diversity among agents and prevent overly deterministic behaviour in the simulation.

### Collectives
In this model, the social network of service provider customers acts as a collective. While this network influences agents’ behaviours, it is not explicitly modelled as an entity with its own state variables or behaviours. Instead, the collective behaviour of the social network emerges from interactions between individual agents, as malicious agents spread misinformation and regular agents communicate and exchange experiences about service providers. This emergent property affects customer decisions on switching service providers or staying, impacting the overall system dynamics. Thus, collectives are modelled implicitly, reflecting how information spreads and affects agent behaviours without directly representing the network.

### Observation
Key outputs of the model are observed by tracking the service provider service selection rates, service levels, opinion values, regret, and the duration and coordination of attacks on service providers over time. Regular agents' behaviour is monitored to assess how often they switch providers based on perceived service availability or negative information, and the opinions they have on service providers, while malicious agents' success is measured by the number of customers they influence through these attacks. The model collects summary statistics on agent behaviours at periodic intervals and examines the variability of agent actions to analyse emergent patterns. There is no explicit use of a “virtual scientist” technique.

## Initialisation
```setup_values.py``` specifies all the parameters inputted into the experiments.
Three *scenario* specific variables can be changed between ```True/False``` to change the experiment type. If all scenarios are ```False```, no malicious agents are introduced in the environment.
- cyberattack = ```True/False``` (if true, then malicious agents can conduct an independent cyberattack, no coordination of attacks)
- misinformation = ```True/False``` (if true, then malicious agents can conduct an independent misinformation campaign, no coordination of attacks)
- coordinated_attack = ```True/False``` (if true, then malicious agents can conduct coordinated cyber and misinformation attacks)

```nAgents``` specifies the number of agents generated, and, depending on the scenario, $\%$ of them are malicious, and others are regular agents. ```nProviders``` specifies the number of service providers in the environment.
Other parameters describe agents' behaviour in the environment.

## Input data
The model does not use input data to represent time-varying processes.

## Other information
### Prerequisites
Python 3.10 or higher version is required.

The following Python libraries are required:
- numpy (version 1.24.2 or higher)
- pandas (version 1.5.3 or higher)
- matplotlib (version 3.7.0 or higher)
- networkx (version 3.0 or higher)

### Run experiments
```
python run.py
```
