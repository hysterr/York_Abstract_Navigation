 #%% Preamble
from Environment import Graph, Prism
from Maps import Risk, Bungalow, LivingArea
from Mission import Mission
from Simulate import Simulation
from copy import deepcopy
from itertools import permutations
import numpy as np

# We will use PRISM to validate paths.
PRISM_PATH = '/Users/jordanhamilton/Documents/PRISM/bin/prism'

#%% ===========================================================================
# Create Environment Objects
# =============================================================================
# Create connections for the environment
risk_matrix = Risk()
connections = Bungalow(risk_matrix)

# Create environment for the agent
num_nodes = max(max(connections))
agent = Graph(n_nodes=num_nodes, n_probs=3) 
agent.Create_Connections(connections)
agent.Create_Map()

#%% ===========================================================================
# Mission Definement
# =============================================================================
agent.dynamics.position = 22 # current position of the robot (node)

# Set a task for the agent using environment nodes.
# agent.mission.tasks = [17, 11, 26, 2, 4, 15, 19, 22]
agent.mission.start = agent.dynamics.position
agent.mission.tasks = [17, 9, 9]
agent.mission.position = 0 # Set the index of the agent's task to 0. 
agent.mission.progress = [agent.mission.tasks[agent.mission.position]]


# Each location along the mission/task will have an intermediate task for the robot to perform
# such as "check" or "hold". The status of each intermediate task is described using "C" and "H"
# and these holders will be used to request the human perform some action when the robot reaches 
# one of the these states.
agent.mission.headers = ['C','C','H']

#%% ===========================================================================
# Create Random Mission
# =============================================================================
# headers = list()
# headers.append("S") # Start sequence node 
# headers.append("C") # Check sequence node (un-ordered)
# headers.append("H") # Hold sequence node (ordered)

agent.Random_Mission(n_nodes=30, hold_rate=0.80, max_unordered=4)

#%% ===========================================================================
# Create Mission
# =============================================================================
mission = Mission(agent)
mission.environment = Graph(n_nodes=num_nodes, n_probs=3)
mission.environment.Create_Connections(mission.connections)
mission.environment.Create_Map()

# Create mission breakdown 
sub_tasks = mission.Breakdown() 		# Create mission breakdown 
sub_tasks = mission.Permute(sub_tasks, apply_end_state=True)	# Create all permutations of the mission
sub_tasks = mission.Solve(sub_tasks)	# Create solution to the mission

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
			agent = Simulation.Select_Path(agent, PRISM_PATH)
		
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


