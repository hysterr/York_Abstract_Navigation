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
# agent.Random_Mission(n_nodes=10, hold_rate=0.90, max_unordered=4)	# Agent does not have all tasks ordered
# human.Random_Mission(n_nodes=10, hold_rate=0.00, max_unordered=1)	# Human is set to have all tasks ordered

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

#%% ===========================================================================
# Simulation
# =============================================================================
# Reset the agent for simulation 
agent = Simulation.Reset(agent)

# Iterate through each stage of the mission
for n_sub_task in range(len(sub_tasks)):
	# Reset mission complete booleans 
	agent.mission.complete = False 
	agent.mission.failed = False
	
	# Update the agent's mission profile
	agent.mission.index = n_sub_task # update sub-mission index for mission profile 
	agent.mission.mission = sub_tasks[agent.mission.index]["Solutions"]["Probability"]["Paths"][0]
	
	# Print dialogue for user
	print("-"*100)
	print(f"Performing Phase {n_sub_task+1}/{len(sub_tasks)} --> {agent.mission.mission}")
	print("-"*100)

	# Run the simulation for the agent until completion.
	while not agent.mission.complete:
		# If the agent has no path, one needs to be created
		if agent.paths.selected.path is None: 
			# Select the path using the simulation class and Select_Path method
			human = Simulation.Select_Path(human, PRISM_PATH, validate=False)

			agent.Update_Heat(connections, path=human.paths.selected.paths, scale=0.5)

			''' ******************************************************************
				Need to find a way to pass the heat map as the path for the 
				selection function and also for the validation as this creates
				a separate map.
				******************************************************************'''

			agent = Simulation.Select_Path(agent, PRISM_PATH, validate=False)
			
		
		# Perform a discrete time-step 
		agent = Simulation.Step(agent)

		# Updating history of the agent 
		# step_data = np.insert(step_data, 0, [n_sub_task+1, agent.mission.position+1, agent.paths.selected.position])
		# agent.dynamics.history = np.vstack((agent.dynamics.history, step_data))

		
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

history = agent.dynamics.history # Initiate history variable for ease

