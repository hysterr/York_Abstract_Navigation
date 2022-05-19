# -*- coding: utf-8 -*-
from Utilities.Prism import Prism
from copy import deepcopy
from random import uniform, randint
import numpy as np

# =============================================================================
# Simulation Class
# =============================================================================
class Simulation:	
	# =============================================================================
	# Reset the Agent 
	# -----------------------------------------------------------------------------
	# Reset the similaton parameters and mission status to perform a new one.
	# =============================================================================
	def Reset(agent):
		agent.paths.history = list() 		# Resets the historic path information
		agent.paths.selected.path = None 	# Resets the current path
		agent.paths.selected.position = 0 	# Resets the position index of the agent
		agent.paths.selected.counter = 0 	# Resets the counter for return states
		
		agent.mission.index = 0 			# Resets the mission index for sub-missions
		agent.mission.position = 0 			# Resets the position of the sub-mission
		agent.mission.failed = False		# Resets the boolean for failed mission
		agent.mission.complete = False		# Resets the boolean for completed mission

		agent.dynamics.position = agent.mission.start
		agent.dynamics.history = np.empty(shape=(0, agent.dynamics.history.shape[1]))

		return agent

	# =============================================================================
	# Step for the Human
	# -----------------------------------------------------------------------------
	# Perform a single discrete step for the human
	#	- Human: human class
	#	- Creativity: Probability that the human will not use the path and will 
	#		instead take a creative route.
	# =============================================================================
	def Step_Human(human, data, creativity=0.05, print_steps=False):
		# The human has three moves of movement: (i) stay at the same location, 
		# (ii) move to the next node along a path, and (iii) move randomly.
		random_movement = uniform(0, 1)

		# Perform random movement if the size of the random_movement variable 
		# is larger than 1 - creativity. 
		if random_movement > 1 - creativity:
			# Human will move randomly to a connected node.
			connecting_nodes = [x for x in human.map[human.dynamics.position]]
					
			# Move in the direction of one of the random nodes
			init_position = human.dynamics.position
			human.dynamics.position = connecting_nodes[randint(0, len(connecting_nodes)-1)]
			if print_steps:
				print(f"\t[{human.mission.events+1}] The human moved from node {init_position} to {human.dynamics.position} (off path)")
			curr_position = init_position
			# next_position = "n/a"

			if len(human.paths.selected.path ) - 1 > human.paths.selected.i_path:
				next_position = human.paths.selected.path[human.paths.selected.i_path+1]
			else:
				next_position = human.paths.selected.path[human.paths.selected.i_path]

			state = "Creative"


		# The human does not move randomly during this step, and instead moves 
		# along a path (if one exists) or stays at the same location (if one does
		# not exist).
		else:
			# Check to see if the human has a mission in the phase...
			if len(human.mission.phase) > 0:

				# The human can only move along the path if it has not already reached the end 
				# of the path. 
				if len(human.paths.selected.path ) - 1 > human.paths.selected.i_path:
					curr_position = human.paths.selected.path[human.paths.selected.i_path]
					next_position = human.paths.selected.path[human.paths.selected.i_path+1]

					# Remove the fisrt element from the human's list.
					human.paths.selected.path.pop(0)

					# Update the position of the human 
					human.dynamics.position = human.paths.selected.path[human.paths.selected.i_path]

					# If the human has an active phase task... check to see if the human reached the target location
					# during this step.	
					if human.dynamics.position == human.mission.phase[human.mission.i_phase]:
						human.mission.phase = []
						human.mission.c_phase = True
						if print_steps:
							print(f"\t[{human.mission.events+1}] The human moved from node {curr_position} and reached the target location {human.dynamics.position}")

					# The human progressed the path. 
					else:
						if print_steps:
							print(f"\t[{human.mission.events+1}] The human moved from node {curr_position} to {next_position} (on path) --> {human.paths.selected.path}")

				state = "Predicted"

			# The human does not move during this step. 
			else:
				curr_position = human.paths.selected.path[human.paths.selected.i_path]
				next_position = human.paths.selected.path[human.paths.selected.i_path]
				if print_steps:
					print(f"\t[{human.mission.events+1}] The human remains at position {human.dynamics.position}") 
				state = "Hold"


		# Data logging for this step of the simulation
		data['position start'] = curr_position
		data['position predict'] = next_position
		data['position final'] = human.dynamics.position
		data['state'] = state
		data['creativity'] = creativity
		data['rand move'] = random_movement
		data['path'] = human.paths.selected.path
		data['Task ID'] = human.mission.t_task,
		data['Phase ID'] = human.mission.i_phase,
		data['Phase '] = human.mission.i_task,
		data['Selected i Path'] = human.paths.selected.i_path,

		return human, data

	# =============================================================================
	# Step for the Agent
	# -----------------------------------------------------------------------------
	# Perform a single discrete step for the agent
	# Inputs: 
	# 	- agent: agent class 
	# 	- map: map that the agent will use for the step. 
	# =============================================================================
	def Step_Agent(agent, data, map, print_steps=False):
		# Create history array for data logging
		history = np.array([
				agent.mission.t_task, 			# Log the index of the missions sub-task
				agent.mission.i_phase,			# Log the index of the phase we are on.
				agent.mission.i_task,			# Log the index of the task in the current phase
				agent.paths.selected.i_path 	# log the current position index along the current path
			])	

		if agent.mission.c_phase is False:
			# Current status of the mission based on the position of the agent.
			curr_position = agent.paths.selected.i_path 			# Current position index for agent along the path
			curr_node = agent.paths.selected.path[curr_position]	# Set current position to be the current index position
			next_node = agent.paths.selected.path[curr_position+1]	# Next task node for agent

			# If the next node is the same location as the current node we do not need to move. 
			# Therefore, check to see if the values are the same and adjust the success rates.
			if curr_node != next_node:
				p_success = map[curr_node][next_node]["Success"] 	# Success probability of the next transition
				p_return  = map[curr_node][next_node]["Return"]	# Return probability of the next transition	
				p_fail    = map[curr_node][next_node]["Fail"]		# Fail probability of the next transition

			# Agent is not moving this step, so set the transition probability of success 
			# to be 1.0 and the failure states to 0.
			else:
				next_node = curr_node	# Set next node to be current node 
				p_success = 1.0			# Set success to 1.0 since we do not need to move.
				p_return  = 0			# Set return to 0.0 since we do not need to move
				p_fail    = 0  			# Set fail to 0.0 since we do not need to move. 

			# Perform movement by creating a random floating value between 0 and 1 and comparing 
			# this value to the success, return and failure probabilities. 
			if p_success == 0 or p_success < 0.90: 
				# The agent holds its position since the selected actions indicates the human is either located at 
				# this node or the probability of success is far too low. 
				agent.mission.n_stuck += 1
				if print_steps:
					print(f"\t[{agent.mission.events+1}] The agent does not move due to human uncertainty... (HOLD)")
				unif = 0
				state = "Hold"

			else:
				# We aren't stuck, reset the counter. 
				agent.mission.n_stuck = 0

				# The agent can move to its intended location... the human is not blocking the way
				total_p = p_success + p_return + p_fail # this value should equal one... (hopefully)
				unif = np.round(uniform(0, total_p), 5)
				if unif <= p_success:
					# The agent successfully moves to the next node
					agent.paths.selected.i_path += 1 		# Update the position counter along the path
					agent.dynamics.position = next_node		# Update the agent's dynamic position 
					agent.paths.selected.n_return = 0		# Reset the return counter

					# Print update to console
					if print_steps:
						print(f"\t[{agent.mission.events+1}] The agent moved from node {curr_node} to {next_node} (Success) --> {agent.paths.selected.path}")
					state = "success"

				elif unif <= (p_success + p_return): 
					# The agent fails to move to the next node and returns to the original node
					agent.paths.selected.n_return += 1

					# if the counter indicates a return on five consecutive attempts end mission. 
					if agent.paths.selected.n_return == 5:
						# agent.mission.complete = True
						agent.mission.failed = True

					# Print update to console.
					if print_steps:
						print(f"\t[{agent.mission.events+1}] The agent returned to node {curr_node} -- counter {agent.paths.selected.n_return} (Return)")
					state = "return"

				else:
					# The agent suffers a failure and the mission should end.
					if print_steps:
						print(f"\t[{agent.mission.events+1}] The agent suffered a catatrophic failure when moving from node {curr_node} to {next_node}	(Fail) ")
					# agent.mission.complete = True
					agent.mission.failed = True
					state = "fail"

			# Check to see and see if the agent has reached the end of the current path.
			# If the agent has reached the end of the current path, the agent needs a new 
			# path to the next waypoint. 
			if (agent.paths.selected.i_path + 1) == len(agent.paths.selected.path):
				# The agent is at the end of the path... but just to be sure.... lets confirm
				if (agent.dynamics.position == agent.paths.selected.path[-1]):
					# YAY we are definitely at the end of the path! 
					# We  	 will append the selected path onto the historic list of paths 
					agent.paths.history.append(agent.paths.selected.path)

					# Reset the selected path variable so during the next time-step a new 
					# path will be created. 
					agent.paths.selected.path = None

					# Increse the task in the phase
					agent.mission.i_task += 1	
					agent.mission.t_task += 1

					# Check to see if the current phase has been completed
					if len(agent.mission.phase) == (agent.mission.i_task):
						# We have reached the end of this phase... 
						agent.mission.c_phase = True
						agent.mission.i_phase += 1

		# If the phase has already been completed by the agent, the agent is most likely waiting for the 
		# human to complete their part of the mission.
		else:
			if print_steps:
				print(f"\t[{agent.mission.events+1}] Agent waits for phase to be completed...")
			curr_node = agent.dynamics.position
			next_node = None
			p_success = 0
			p_return = 0
			p_fail = 0
			unif = None
						

		# Append to the data dictionary
		data['position start'] = curr_node
		data['position ideal'] = next_node
		data['position final'] = agent.dynamics.position
		data['success'] = [p_success, 0, p_success]
		data['return'] = [p_return, p_success, p_success+p_return]
		data['fail'] = [p_fail, p_success+p_return, p_success + p_return + p_fail]
		data['probability'] = unif
		data['state'] = state
		data['path'] = agent.paths.selected.path
		data['path index'] = agent.paths.selected.i_path
		data['Task ID'] = agent.mission.t_task
		data['Phase Number'] = agent.mission.i_phase
		data['Phase Task'] = agent.mission.i_task

		# Create a history array which will be appended to the history at the end 
		# of the current step. 
		# history = np.empty(shape=(0, agent.dynamics.history.shape[1]))
		# history = np.array([curr_node, next_node, agent.dynamics.position, p_success, p_success+p_return, unif])
		history = np.append(history, 
			[
					curr_node,					# Current node location
					next_node, 					# Next node location in the path
					agent.dynamics.position, 	# Final position of the agent after the step
					p_success, 					# Probability of success for this step
					p_success+p_return, 		# Probability of return for this step
					p_success+p_return+p_fail,	# Probability of fail for this step
					unif 						# Uniform value used for step simulation
			])


		# Update the history of the agent to the dynamics class 
		agent.dynamics.history = np.vstack((agent.dynamics.history, history))					

		return agent, data

	# =============================================================================
	# Human Request Redirection
	# -----------------------------------------------------------------------------
	# The method "Human_Redirect" occurs when the human is blocking the robot's 
	# path or task node, with a request given to the human to redirect to a safe 
	# location in the environment. 
	# =============================================================================
	def Human_Redirect(agent, safe_locations):
		# Use the task list
		tasks = agent.mission.phase
		task_ID = agent.mission.i_task

		# We only want to prevent safe locatiosn being used in future tasks, so use 
		# the current task ID to look into the future, not the past. 
		remaining_tasks = tasks[task_ID:]

		# We also do not want the agent to ask for redirection to a node which exists 
		# on the agent's current path.
		for p_task in agent.paths.selected.path:
			if p_task not in remaining_tasks:
				remaining_tasks.append(p_task)

		# Remove a location from the safe_locations list if the location is unavailable 
		# due to it being present in the remaining_tasks variable. 
		for l in reversed(safe_locations):
			if l in remaining_tasks:
				safe_locations.remove(l)

		# If the number of safe location is zero, that means we cannot issue a redirect.
		# in that scenario, we should ask the human to move to a location which isn't 
		# identified as being safe, but does not obstruct the operation of the robot 
		if len(safe_locations) == 0:
			# Create a list of all locations in the environment 
			avail_locations = [i for i in range(1, len(agent.map)+1)]

			# remove all conflicting locations 
			for l in reversed(avail_locations):
				if l in remaining_tasks:
					avail_locations.remove(l)

			# Select a random node based on the remaining available locations
			selected_node = avail_locations[randint(0, len(avail_locations)-1)]

		else:
			# Randomly select a location from the remaining identified safe locations
			selected_node = safe_locations[randint(0, len(safe_locations)-1)]

		return selected_node			


	# =============================================================================
	# Select Path
	# -----------------------------------------------------------------------------
	# The method "Select_Path" identifies the agent's location relative to the task 
	# and mission and creates a path to the next waypoint. 
	#
	# Paths are stored within the agent's path class (agent.paths) and are selected 
	# based on a PRISM validation analysis. 
	# =============================================================================
	def Select_Path(entity, prism_path=None, validate=True, heated=False, print_output=True):
		# We have two classes of agents ("agent" and "human") which require different 
		# processes.
		if entity.ID == "Human":
			curr_position = entity.dynamics.position
			next_waypoint = entity.mission.phase[entity.mission.i_task]
			
			# Find the path of least distance
			entity = entity.Dijkstra(curr_position, next_waypoint, entity.paths.min_dist, method="Distance")
			entity.paths.selected = deepcopy(entity.paths.min_dist)

		if entity.ID == "Agent":
			curr_position = entity.dynamics.position
			next_waypoint = entity.mission.phase[entity.mission.i_task]
			# For the agent we find two solutions: least distance and highest prob of success
			if heated is False:
				# Use the heated map for path finding
				entity = entity.Dijkstra(curr_position, next_waypoint, entity.paths.min_dist, method="Distance")
				entity = entity.Dijkstra(curr_position, next_waypoint, entity.paths.max_prob, method="Probability")	

			elif heated is True:
				# Use the heated map for path finding
				entity = entity.Dijkstra(curr_position, next_waypoint, entity.paths.min_dist, method="Distance",    map=entity.heat_map)
				entity = entity.Dijkstra(curr_position, next_waypoint, entity.paths.max_prob, method="Probability", map=entity.heat_map)	

			if validate and prism_path is not None: 
				# select the path through validation
				entity = Simulation.__Validate(entity, prism_path)

			else:
				# select the highest probability path
				entity.paths.selected = deepcopy(entity.paths.max_prob)

			entity.paths.selected.i_path = 0

		# Based on the path distance, compute the estimated completion time based on the agent's speed
		entity.paths.selected.time = entity.paths.selected.length / entity.dynamics.velocity

		# Create a distance vector where each entry in the vector is a cumulative distance 
		# value from the previous node. 
		entity = Simulation.Path_Cummulative_Distance(entity)

		# If the curr_position == next_waypoint, the agent will remain in the same location
		# but we still want to evaluate the path. 
		if curr_position == next_waypoint:
			entity.paths.selected.path.append(entity.paths.selected.path[0])

		if print_output is True:
			print(f"The {entity.ID} begins task {entity.mission.t_task+1} and will path from node {curr_position} to node: {next_waypoint} using path {entity.paths.selected.path}")

		return entity

	# =============================================================================
	# Validate the Path
	# -----------------------------------------------------------------------------
	# When creating path using the Select_Path method, the path may require valida-
	# -ation using PRISM. This method performs that action and selects the path 
	# which has the highest validated probability of success.
	# =============================================================================
	def __Validate(agent, prism_path):
		# Positions for validation
		curr_position = agent.dynamics.position
		next_waypoint = agent.mission.phase[agent.mission.i_task]
		
		export_file_name = "Prism/Model_1.prism" # Prism folder for exporting the models

		# Since the path has yet to be validated, we should analyse both paths 
		# using PRISM and select the path which has the best validated probability 
		# of success. So... create the first action set. 
		action_1 = Prism.Generate_Action(agent.map, num_solutions=1, initial_guess=agent.paths.min_dist.path)
		action_2 = Prism.Generate_Action(agent.map, num_solutions=1, initial_guess=agent.paths.max_prob.path)

		# Run PRISM validation on the 1st path 
		code = Prism.Create_Model(agent.map, curr_position, next_waypoint, action_1[0,:])
		file_path, model_name = Prism.Export_Model(code, file_name=export_file_name)
		agent.paths.min_dist.valid = Prism.Simulate(prism_path, file_path+model_name, output_files=True)
		    
		# Run PRISM validation on the 2nd path
		code = Prism.Create_Model(agent.map, curr_position, next_waypoint, action_2[0,:])
		file_path, model_name = Prism.Export_Model(code, file_name=export_file_name)
		agent.paths.max_prob.valid = Prism.Simulate(prism_path, file_path+model_name, output_files=True)

		# Select the path based on validation probability as the PCTL relationship for 
		# PRISM will return the maximum probability value. Therefore...
		dec_place_round = 5 # Round to prevent infinite rounding errors...

		# If the validation for the min_dists is equal or greater than the validation for 
		# max_prob, then we should use the shortest distance path...
		if np.round(agent.paths.min_dist.valid, dec_place_round) >= np.round(agent.paths.max_prob.valid, dec_place_round):
			agent.paths.selected = deepcopy(agent.paths.min_dist)	# Deep copy the min_dist

		# If they aren't, we should use the path with the highest probability.
		else:
			agent.paths.selected = deepcopy(agent.paths.max_prob)	# Deep copy the max_prob

		return agent


	# =============================================================================
	# Path Cumulative Distance
	# -----------------------------------------------------------------------------
	# Since a path between two nodes occurs with movement (potentially) between 
	# multiple intermediate nodes, the distance can be measured as relative lengths
	# between multiple points 
	# 
	#                   l1         l2          l3
	# nodes        n1 ------> n2 ------> n3 ------> n4
	# dists             3          2           4
	# cum dists         3          5           9
	# =============================================================================
	def Path_Cummulative_Distance(agent):
		# Create a cumulative distance vector for the path so we can keep track of the 
		# robot's progress along the path.
		agent.paths.selected.dist_cum = list() # Reset the cummulative distance of the selected path.
		cum_dist = 0 # set the cumulative distance variable 

		# Iterate over each node in the selected path variable
		for node in range(1, len(agent.paths.selected.path)):
			x1 = agent.paths.selected.path[node-1] # previous node
			x2 = agent.paths.selected.path[node]   # current node

			# Add the distance to the next node based on the environment map. 
			cum_dist += agent.map[x1][x2]["Distance"]

			# Append the distance to the cumulative distance 
			agent.paths.selected.dist_cum.append(cum_dist) 


		return agent
	