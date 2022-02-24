#%% ===========================================================================
# Preamble
# =============================================================================
from Utilities.Environment import Graph, Prism
from Utilities.Maps import Risk, Bungalow, LivingArea
from Utilities.Mission import Mission
from Utilities.Simulate import Simulation
from copy import deepcopy
from itertools import permutations
import numpy as np
import pandas as pd

PRISM_PATH = '/Users/jordanhamilton/Documents/PRISM/bin/prism'

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
human.dynamics.position = 24 # current position of the robot (node)/
human.mission.start = human.dynamics.position

# agent.Random_Mission(n_nodes=10, phase_rate=0.80, max_unordered=4, human_rate=0.30, max_human=1)	# Agent does not have all tasks ordered

agent.dynamics.position = 22 # current position of the robot (node)
agent.mission.start = agent.dynamics.position

agent.mission.tasks = [26, 11, 15, 4, 21]
agent.mission.headers = ['U', 'U', 'H', 'U', 'O']

agent.mission.position = 0 # Set the index of the agent's task to 0. 
agent.mission.progress = [agent.mission.tasks[agent.mission.position]]

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

#%% ===========================================================================
# Simulation
# =============================================================================
# Reset the agent for simulation 
agent = Simulation.Reset(agent)

# Reset mission complete booleans 
agent.mission.complete = False 
agent.mission.failed = False
human.mission.c_phase = True

agent.mission.complete = False

# Start the simulation inside a while loop
while agent.mission.complete is False:

	# If the c_phase boolean is True, that indicates a new phase will be started if one exists.
	if agent.mission.c_phase is True and human.mission.c_phase is True: 
		# Set the mission phase for the agent
		agent.mission.phase = agent.mission.breakdown[agent.mission.i_phase-1]['Solutions']['Probability']['Paths'][0]

		# Set the mission phase for the human
		human.mission.phase = agent.mission.breakdown[agent.mission.i_phase-1]["H"]
	
		# Reset the task index and complete boolean
		agent.mission.i_task = 1 	  
		agent.mission.c_phase = False 

		# Print statement for phase console information
		print("-"*100)
		print(f"Performing Phase {agent.mission.i_phase}/{agent.mission.n_phase} --> Agent Tasks: {agent.mission.phase} --- Human Tasks: {human.mission.phase}")
		print("-"*100)

		# If the human has a task to be performed in this phase, the length of the phase 
		# will be greater than zero, and therefore we can start to produce a path for this 
		# human's mission.
		if len(human.mission.phase) > 0:
			human.paths.selected.path = None
			human.mission.c_phase = False
			print(f"Requesting the human performs task at node {human.mission.phase[0]}")

	# Create a path for the human if one does not exist. A path is created normally if the 
	# agent has a task to perform in the current phase, otherwise a path is created artificially 
	# by keeping the human at the same location.
	if human.paths.selected.path is None or human.paths.selected.off_path is True:
		if len(human.mission.phase) > 0:
			human = Simulation.Select_Path(human, PRISM_PATH, validate=False, heated=False)
			human.paths.selected.off_path = False # Reset the off path trigger
		else:
			human.paths.selected.path = [human.dynamics.position, human.dynamics.position]

	# Check to see if the agent has a path selected, and create one if it doesn't.
	if agent.paths.selected.path is None and agent.mission.c_phase is False:
		agent.Update_Heat(human.paths.selected.path, human.dynamics.position)
		agent = Simulation.Select_Path(agent, PRISM_PATH, validate=False, heated=True)

	# Update the heat map for the agent for this discrete step based
	agent.Update_Heat(human.paths.selected.path, human.dynamics.position)

	# Perform a discrete step along the current path.
	human = Simulation.Step_Human(human, creativity=0.05)
	agent = Simulation.Step_Agent(agent, map=agent.heat_map)
	
	agent.mission.events += 1
	human.mission.events += 1

	# # If the human has an active phase task... check to see if the human reached the target location
	# # during this step.			
	# if len(human.mission.phase) > 0 and human.dynamics.position == human.mission.phase[human.mission.i_phase]:	
	# 	human.mission.phase = []
	# 	human.mission.c_phase = True
	# 	print(f"\t[{human.mission.events+1}] The human reached the target location {human.dynamics.position}")

	# Check to see if the mission has been completed based on the number of phases 
	# that have been completed.
	if agent.mission.i_phase > agent.mission.n_phase:
		agent.mission.complete = True

	# If the agent suffered a failure during the step, end the mission.
	if agent.mission.failed is True:
		break

if agent.mission.failed is True:
	print("-"*100)
	print("Agent failed the mission.")
	print("-"*100)
else:
	print("-"*100)
	print("Agent completed the mission.")
	print("-"*100)

history_agent = pd.DataFrame(agent.dynamics.history, columns=agent.dynamics.history_columns)
history_human = pd.DataFrame(human.dynamics.history, columns=human.dynamics.history_columns)


# Options for printing to the console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# history = agent.dynamics.history # Initiate history variable for ease
# df = pd.DataFrame(history)#, columns = ['Column_A','Column_B','Column_C'])



