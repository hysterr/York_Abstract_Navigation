# =============================================================================
# Preamble
# =============================================================================
from Utilities.Environment import Graph, Prism
from Utilities.Maps import Risk, Bungalow, LivingArea
from Utilities.Mission import Mission
from Utilities.Simulate import Simulation
from copy import deepcopy
from itertools import permutations
import numpy as np

PRISM_PATH = '/Users/jordanhamilton/Documents/PRISM/bin/prism'

# =============================================================================
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

# =============================================================================
# Mission Definement
# =============================================================================
agent.Random_Mission(n_nodes=10, hold_rate=0.90, max_unordered=4)	# Agent does not have all tasks ordered
# human.Random_Mission(n_nodes=10, hold_rate=0.00, max_unordered=1)	# Human is set to have all tasks ordered

# agent.dynamics.position = 22 # current position of the robot (node)
# agent.mission.start = agent.dynamics.position
# agent.mission.tasks = [17, 9, 13]
# agent.mission.headers = ['H', 'H', 'H']
# agent.mission.position = 0 # Set the index of the agent's task to 0. 
# agent.mission.progress = [agent.mission.tasks[agent.mission.position]]

human.dynamics.position = 30 # current position of the robot (node)
human.mission.start = human.dynamics.position
human.mission.tasks = [20, 11, 16]
human.mission.headers = ['H', 'H', 'H']
human.mission.position = 0 # Set the index of the agent's task to 0. 
human.mission.progress = [human.mission.tasks[human.mission.position]]

# =============================================================================
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


#%% ===========================================================================
# Simulation
# =============================================================================
# Reset the agent for simulation 
agent = Simulation.Reset(agent)

# Reset mission complete booleans 
agent.mission.complete = False 
agent.mission.failed = False

# Start the simulation inside a while loop
while agent.mission.complete is False:

	# Identify the current phase of the mission and introduce new phases 
	# into the current agenda. 
	if agent.mission.c_phase is True: 
		agent.mission.phase = agent.mission.breakdown[agent.mission.i_phase-1]['Solutions']['Probability']['Paths'][0]
		agent.mission.i_task = 1
		agent.mission.c_phase = False
		print("-"*100)
		print(f"Performing Phase {agent.mission.i_phase}/{agent.mission.n_phase} --> {agent.mission.phase}")
		print("-"*100)

	# if the agent has no path, one needs to be created.
	if agent.paths.selected.path is None:
		agent = Simulation.Select_Path(agent, PRISM_PATH, validate=False)

	# Perform a disctete step along the current path.
	agent = Simulation.Step(agent)

	# # If we have reached the end of the current phase...
	if agent.mission.i_phase > agent.mission.n_phase:
		agent.mission.complete = True


	# If the agent suffered a failure during the step, end the mission.
	if agent.mission.failed is True:
		break


if agent.mission.complete is True: 
	if agent.mission.failed is True:
		print("-"*100)
		print("Agent failed the mission.")
		print("-"*100)
	else:
		print("-"*100)
		print("Agent completed the mission.")
		print("-"*100)


# history = agent.dynamics.history # Initiate history variable for ease

