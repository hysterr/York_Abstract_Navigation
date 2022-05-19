# %% ===========================================================================
# Preamble
# =============================================================================
from Utilities.Environment import Graph
from Utilities.Prism import Prism
from Utilities.Maps import Risk, Bungalow, LivingArea
from Utilities.Mission import Mission, Preset_Missions
from Utilities.Simulate import Simulation
from copy import deepcopy
from itertools import permutations
from random import randint
from alive_progress import alive_bar
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import os
import glob
import pickle

PRISM_PATH = '/Users/jordanhamilton/Documents/PRISM/bin/prism'

#%% ===========================================================================
# CLI Input Arguments
# =============================================================================
# Input argument for running a simulation
if sys.argv[1] == 'True':
	Run_Sim = True
else:
	Run_Sim = False

#%% ===========================================================================
# Similation specific parameters 
# =============================================================================
N_SIMS = 1

# Each time a new simulation starts, a default print statement with the mission 
# overview is outputted to the console. 
print_mission_overview = True

# Each time a path is produced, a detailed output as they occur will be produced
print_paths_agent = False
print_paths_human = False

# Print steps - print a detailed output of each step as they occurr in simulation.
print_steps_agent = True
print_steps_human = True

PRISM_path_validation_human = False
PRISM_path_validation_agent = False

#%% ===========================================================================
# Create Environment Objects
# =============================================================================
risk_matrix = Risk()
connections, safe_locations = Bungalow(risk_matrix)

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
human.dynamics.position = 8 #randint(1, num_nodes) # current position of the robot (node)
human.mission.start = human.dynamics.position
human_creativity = 0.00

# Load preset missions 
agent.dynamics.position = 1# randint(1, num_nodes)
agent.mission.start = agent.dynamics.position

tasks, headers = Preset_Missions.Mission_One(start=agent.mission.start, final=22)
agent.mission.tasks = tasks
agent.mission.headers = headers
agent.mission.position = 0
agent.mission.progress = [agent.mission.tasks[agent.mission.position]]


# agent.Random_Mission(n_nodes=10, phase_rate=0.80, max_unordered=4, human_rate=0.30, max_human=1)	# Agent does not have all tasks ordered

# agent.dynamics.position = 22 # current position of the robot (node)
# agent.mission.start = agent.dynamics.position

# agent.mission.tasks   = [ 26,  11,  8,    4,  21]
# agent.mission.headers = ['U', 'U', 'H', 'U', 'O']

# # agent.mission.tasks = []

# agent.mission.position = 0 # Set the index of the agent's task to 0. 
# agent.mission.progress = [agent.mission.tasks[agent.mission.position]]

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

data = dict()
prev_state = None

if Run_Sim is True:
	simulation_steps = 0
	agent.mission.complete = False

	# Start the simulation inside a while loop
	while agent.mission.complete is False:
		simulation_steps += 1

		# Initialise dictionary for this step of the simulation
		data[simulation_steps] = dict()
		data[simulation_steps]["human"] = dict()
		data[simulation_steps]["agent"] = dict()

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
			if print_mission_overview:
				print("-"*100)
				print(f"Performing Phase {agent.mission.i_phase}/{agent.mission.n_phase} --> Agent Tasks: {agent.mission.phase} --- Human Tasks: {human.mission.phase}")
				print("-"*100)

			# If the human has a task to be performed in this phase, the length of the phase 
			# will be greater than zero, and therefore we can start to produce a path for this 
			# human's mission.
			if len(human.mission.phase) > 0:
				human.paths.selected.path = None
				human.mission.c_phase = False
				if print_mission_overview:
					print(f"Requesting the human performs task at node {human.mission.phase[0]}")

		# Create path for the human 
		if len(human.mission.phase) > 0:
			human = Simulation.Select_Path(human, PRISM_PATH, validate=PRISM_path_validation_human, heated=False, print_output=print_paths_human)
		else:
			human.paths.selected.path = [human.dynamics.position, human.dynamics.position]

		# Create path for the agent 
		agent.Update_Heat(human)
		agent = Simulation.Select_Path(agent, PRISM_PATH, validate=PRISM_path_validation_agent, heated=True, print_output=print_paths_agent)

		# Perform a discrete step along the current path.
		human, data[simulation_steps]['human'] = Simulation.Step_Human(human, data[simulation_steps]['human'], print_steps=print_steps_human, creativity=human_creativity)
		agent, data[simulation_steps]['agent'] = Simulation.Step_Agent(agent, data[simulation_steps]['agent'], print_steps=print_steps_agent, map=agent.heat_map)

		# Perform a check to see if the robot is stuck due to the human blocking the path
		if agent.mission.n_stuck > 1:
			# Request the human moves to a safer location to allow the robot to continue with the mission
			# We should only request the human moves if the human has no active phase tasks
			if human.mission.c_phase is True and prev_state != "Redirect":
				# Request movement to a safe location 
				redirect_location = Simulation.Human_Redirect(agent, safe_locations)
				human.mission.phase = [redirect_location]
				human.mission.c_phase = False
				data[simulation_steps]['agent']['state'] = 'Redirect'

				if print_steps_agent:
					print(f"\t\tRequesting the human redirects to node {redirect_location}")

			if agent.mission.n_stuck >= 10:
				print("-"*100)
				print(f"The agent has been stuck in location {agent.dynamics.position} for {agent.mission.n_stuck} steps with the human located at {human.dynamics.position}. Mission will end.")
				agent.mission.failed = True
				
		# To prevent multiple redirect states from being performed consecutively, update the prev_state 
		# so the next step has a better understanding of what happened previouly. 
		prev_state = data[simulation_steps]['agent']['state']

		agent.mission.events += 1
		human.mission.events += 1

		# Check to see if the mission has been completed based on the number of phases 
		# that have been completed.
		if agent.mission.i_phase > agent.mission.n_phase:
			agent.mission.complete = True

		# If the agent suffered a failure during the step, end the mission.
		if agent.mission.failed is True:
			break

	if agent.mission.failed is True:
		if print_mission_overview:
			print("-"*100)
			print("Agent failed the mission.")
			print("-"*100)
	else:
		if print_mission_overview:
			print("-"*100)
			print("Agent completed the mission.")
			print("-"*100)


columns = ["Step", "RPos 1", "RPos X", "RPos 2", "P_s", "P_r", "P_f", "P", "State", "HPos 1", "HPos X", "HPos 2", "State"]
data_array = np.empty(shape=(len(data), len(columns)), dtype='<U21')

# Compile data array 
for i in range(len(data)):
	new_data = np.array([
		i+1,										# Stepfile_p
		data[i+1]['agent']['position start'],		# Robot Position 1
		data[i+1]['agent']['position ideal'],		# Robot Position X
		data[i+1]['agent']['position final'],		# Robot Position 2
		data[i+1]['agent']['success'][0],			# Robot success value
		data[i+1]['agent']['return'][0],			# Robot success value
		data[i+1]['agent']['fail'][0],				# Robot success value
		data[i+1]['agent']['probability'],			#
		data[i+1]['agent']['state'],				# Robot State
		data[i+1]['human']['position start'],		# Human Position 1
		data[i+1]['human']['position predict'],		# Human Posiiton Predicton
		data[i+1]['human']['position final'],		# Human Position 2
		data[i+1]['human']['state']					# Human State
	])

	data_array[i,:] = new_data


# Compile into a dataframe
history = pd.DataFrame(data_array, columns=columns)
















