from Utilities.Environment import Graph, Prism
from Utilities.Maps import Risk, Bungalow, LivingArea
from Utilities.Mission import Mission
from Utilities.Simulate import Simulation
from copy import deepcopy
from itertools import permutations
import numpy as np
import pandas as pd

#%% ===========================================================================
# Create Environment Objects
# =============================================================================
risk_matrix = Risk()
connections = Bungalow(risk_matrix)

# Create environment for the agent
num_nodes = max(max(connections))
agent = Graph(n_nodes=num_nodes, ID="Agent", n_probs=3) 
agent.Create_Connections(connections)
agent.Create_Map()

# Create environment for the human 
human = Graph(n_nodes=num_nodes, ID="Human", n_probs=2)
human.Create_Connections(connections)
human.Create_Map(agent.map)

#%% ===========================================================================
# Mission Definement
# =============================================================================
agent.dynamics.position = 22 # current position of the robot (node)
agent.mission.start = agent.dynamics.position
agent.mission.tasks = [17, 9, 13]
agent.mission.headers = ['H', 'H', 'H']
agent.mission.position = 0 # Set the index of the agent's task to 0. 
agent.mission.progress = [agent.mission.tasks[agent.mission.position]]

human.dynamics.position = 30 # current position of the robot (node)
human.mission.start = human.dynamics.position
human.mission.tasks = [20, 11, 16]
human.mission.headers = ['H', 'H', 'H']
human.mission.position = 0 # Set the index of the agent's task to 0. 
human.mission.p/rogress = [human.mission.tasks[human.mission.position]]

#%% ===========================================================================
# Mission Breakdown
# =============================================================================
mission = Mission(agent)
mission.environment = Graph(n_nodes=num_nodes, ID="Agent", n_probs=3)
mission.environment.Create_Connections(mission.connections)
mission.environment.Create_Map()

sub_tasks = mission.Breakdown()
sub_tasks = mission.Permute(sub_tasks, apply_end_state=True)
sub_tasks = mission.Solve(sub_tasks)

# Compile the mission plan
agent.Compile_Mission(sub_tasks)