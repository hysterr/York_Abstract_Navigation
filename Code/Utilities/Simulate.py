# -*- coding: utf-8 -*-
from Utilities.Environment import Prism
from copy import deepcopy
from random import uniform
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
	# Step
	# -----------------------------------------------------------------------------
	# Perform a single discrete step	
	# =============================================================================
	def Step_Agent(agent, map):
		# Create history array for data logging
		history = np.array([
				agent.mission.t_task, 			# Log the index of the missions sub-task
				agent.mission.i_phase,			# Log the index of the phase we are on.
				agent.mission.i_task,			# Log the index of the task in the current phase
				agent.paths.selected.i_path 	# log the current position index along the current path
			])	

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

		else:
			next_node = curr_node	# Set next node to be current node 
			p_success = 1.0			# Set success to 1.0 since we do not need to move.
			p_return  = 0			# Set return to 0.0 since we do not need to move
			p_fail    = 0  			# Set fail to 0.0 since we do not need to move. 

		# Perform movement by creating a random floating value between 0 and 1 and comparing 
		# this value to the success, return and failure probabilities. 
		total_p = p_success + p_return + p_fail
		unif = uniform(0, total_p) 
		if unif <= p_success:
			# The agent successfully moves to the next node
			agent.paths.selected.i_path += 1 		# Update the position counter along the path
			agent.dynamics.position = next_node		# Update the agent's dynamic position 
			agent.paths.selected.n_return = 0		# Reset the return counter

			# Print update to console
			print(f"\tThe agent moved from node {curr_node} to {next_node} (Success)")
		
		elif unif <= (p_success + p_return): 
			# The agent fails to move to the next node and returns to the original node
			agent.paths.selected.n_return += 1

			# if the counter indicates a return on five consecutive attempts end mission. 
			if agent.paths.selected.n_return == 5:
				# agent.mission.complete = True
				agent.mission.failed = True

			# Print update to console.
			print(f"\tThe agent returned to node {curr_node} -- counter {agent.paths.selected.n_return} (Return)")

		else:
			# The agent suffers a failure and the mission should end.
			print(f"\tThe agent suffered a catatrophic failure when moving from node {curr_node} to {next_node}	(Fail) ")
			# agent.mission.complete = True
			agent.mission.failed = True


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

		return agent

	# =============================================================================
	# Select Path
	# -----------------------------------------------------------------------------
	# The method "Select_Path" identifies the agent's location relative to the task 
	# and mission and creates a path to the next waypoint. 
	#
	# Paths are stored within the agent's path class (agent.paths) and are selected 
	# based on a PRISM validation analysis. 
	# =============================================================================
	def Select_Path(agent, prism_path=None, validate=True, heated=False):
		# We have two classes of agents ("agent" and "human") which require different 
		# processes.
		if agent.ID == "Human":
			curr_position = agent.dynamics.position
			next_waypoint = agent.mission.phase[agent.mission.i_task]
			
			# Find the path of least distance
			agent = agent.Dijkstra(curr_position, next_waypoint, agent.paths.min_dist, method="Distance")
			agent.paths.selected = deepcopy(agent.paths.min_dist)


		if agent.ID == "Agent":
			curr_position = agent.dynamics.position
			next_waypoint = agent.mission.phase[agent.mission.i_task]
			# For the agent we find two solutions: least distance and highest prob of success
			if heated is False:
				# Use the heated map for path finding
				agent = agent.Dijkstra(curr_position, next_waypoint, agent.paths.min_dist, method="Distance")
				agent = agent.Dijkstra(curr_position, next_waypoint, agent.paths.max_prob, method="Probability")	

			elif heated is True:
				# Use the heated map for path finding
				agent = agent.Dijkstra(curr_position, next_waypoint, agent.paths.min_dist, method="Distance",    map=agent.heat_map)
				agent = agent.Dijkstra(curr_position, next_waypoint, agent.paths.max_prob, method="Probability", map=agent.heat_map)	

			if validate and prism_path is not None: 
				# select the path through validation
				agent = Simulation.__Validate(agent, prism_path)

			else:
				# select the highest probability path
				agent.paths.selected = deepcopy(agent.paths.max_prob)

		# Based on the path distance, compute the estimated completion time based on the agent's speed
		agent.paths.selected.time = agent.paths.selected.length / agent.dynamics.velocity

		# Create a distance vector where each entry in the vector is a cumulative distance 
		# value from the previous node. 
		agent = Simulation.Path_Cummulative_Distance(agent)

		# If the curr_position == next_waypoint, the agent will remain in the same location
		# but we still want to evaluate the path. 
		if curr_position == next_waypoint:
			agent.paths.selected.path.append(agent.paths.selected.path[0])

		print(f"The {agent.ID} begins task {agent.mission.t_task+1} and will path from node {curr_position} to node: {next_waypoint} using path {agent.paths.selected.path}")

		return agent

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
	# dists             3          2         4
	# cum dists         3          5         9
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
	