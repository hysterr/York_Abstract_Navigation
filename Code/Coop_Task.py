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
# agent.Random_Mission(n_nodes=5, phase_rate=1.00, max_unordered=4, human_rate=0.30, max_human=1)	# Agent does not have all tasks ordered

agent.dynamics.position = 22 # current position of the robot (node)
agent.mission.start = agent.dynamics.position


agent.mission.tasks = [26, 11, 15, 4, 21]
agent.mission.headers = ['U', 'U', 'H', 'U', 'O']

# agent.mission.tasks = [26, 11, 4, 21]
# agent.mission.headers = ['U', 'U', 'U', 'O']

agent.mission.position = 0 # Set the index of the agent's task to 0. 
agent.mission.progress = [agent.mission.tasks[agent.mission.position]]

human.dynamics.position = 20 # current position of the robot (node)
human.mission.start = human.dynamics.position

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

# Start the simulation inside a while loop
while agent.mission.complete is False:

	# Identify the current phase of the mission and introduce new phases 
	# into the current agenda. 
	if agent.mission.c_phase is True: 
		# Set the current phase
		agent.mission.phase = agent.mission.breakdown[agent.mission.i_phase-1]['Solutions']['Probability']['Paths'][0]

		# Update the human task list for this phase
		human.mission.phase = agent.mission.breakdown[agent.mission.i_phase-1]["H"]

		agent.mission.i_task = 1 	  # Reset the task index value 
		agent.mission.c_phase = False # Reset the complete boolean
		print("-"*100)
		print(f"Performing Phase {agent.mission.i_phase}/{agent.mission.n_phase} --> {agent.mission.phase}")
		print("-"*100)

		# We need to identify whether in the current phase, the human has a task.
		if len(human.mission.phase) > 0:
			# Check to see if we have predicted a path for the human.
			if human.paths.selected.path is None:
				print(f"Requesting the human performs task {human.mission.phase[0]}")
				agent, human = Simulation.Select_Path(agent, PRISM_PATH, validate=False, human=human)


	# If the agent reaches the end of path, the path is set to None, triggering 
	# a path to be created for the next task/waypoint.
	if agent.paths.selected.path is None:
		agent, human = Simulation.Select_Path(agent, PRISM_PATH, validate=False, human=human)

	# Perform a disctete step along the current path.
	agent = Simulation.Step(agent)
	

	# # If we have reached the end of the current phase...
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

history = pd.DataFrame(agent.dynamics.history, columns=agent.dynamics.history_columns)

# Options for printing to the console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# history = agent.dynamics.history # Initiate history variable for ease
# df = pd.DataFrame(history)#, columns = ['Column_A','Column_B','Column_C'])

