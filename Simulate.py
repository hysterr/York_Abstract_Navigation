# -*- coding: utf-8 -*-
from Environment import Prism
from copy import deepcopy
from random import uniform

# =============================================================================
# Simulation Class
# =============================================================================
class Simulation: 
	
	# =============================================================================
	# Reset the Agent 
	# -----------------------------------------------------------------------------
	# Reset the similaton to perform a new one.
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

		return agent

	# =============================================================================
	# Step
	# -----------------------------------------------------------------------------
	# Perform a single discrete step	
	# =============================================================================
	def Step(agent):
		curr_position = agent.paths.selected.position 			# Current position index for agent
		curr_node = agent.paths.selected.path[curr_position]	# Current task node for agent
		next_node = agent.paths.selected.path[curr_position+1]	# Next task node for agent

		p_success = agent.map[curr_node][next_node]["Success"] 	# Success probability of the next transition
		p_return = agent.map[curr_node][next_node]["Return"]	# Return probability of the next transition	
		p_fail = agent.map[curr_node][next_node]["Fail"]		# Fail probability of the next transition

		unif = uniform(0, 1) # Draw a random number from a uniform distribution

		if unif <= p_success:
			# The agent successfully moves to the next node
			agent.paths.selected.position += 1 		# Update the position counter
			agent.dynamics.position = next_node		# Update the agent's dynamic position 
			agent.paths.selected.counter = 0		# Reset the return counter

			# Print update to console
			print(f"\tThe agent moved from node {curr_node} to {next_node} (Success)")
		
		elif unif <= (p_success + p_return): 
			# The agent fails to move to the next node and returns to the original node
			agent.paths.selected.counter += 1

			# if the counter indicates a return on five consecutive attempts end mission. 
			if agent.paths.selected.counter == 5:
				agent.mission.complete = True
				agent.mission.failed = True

			# Print update to console.
			print(f"\tThe agent returned to node {curr_node} -- counter {agent.paths.selected.counter} (Return)")

		else:
			# The agent suffers a failure and the mission should end.
			print(f"\tThe agent suffered a catatrophic failure when moving from node {curr_node} to {next_node}	(Fail) ")
			agent.mission.complete = True
			agent.mission.failed = True


		# Check to see and see if the agent has reached the end of the currnet path.
		# If the agent has reached the end of the current path, the agent needs a new 
		# path to the next waypoint. 
		if (agent.paths.selected.position + 1) == len(agent.paths.selected.path):
			# The agent is at the end of the path... but just to be sure.... lets 
			# make sure! 
			if (agent.dynamics.position == agent.paths.selected.path[-1]):
				# YAY we are definitely at the end of the path! 
				# We will append the selected path onto the historic list of paths 
				agent.paths.history.append(agent.paths.selected.path)

				# Reset the selected path variable so during the next time-step a new 
				# path will be created. 
				agent.paths.selected.path = None

				# Increase the mission counter position 
				agent.mission.position += 1

				# Check to see if the mission has been completed.
				if len(agent.mission.mission) == (agent.mission.position + 1):
					# We have reached the end of this mission... 
					agent.mission.complete = True
					agent.mission.position = 0 # Reset the mission position for the next sub-mission
					

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
	def Select_Path(agent, prism_path=None):
		# We will use PRISM to validate paths.
		if prism_path is None:
			prism_path = '/Users/jordanhamilton/Documents/PRISM/bin/prism'
		
		# We need to first create a path for the agent between the current location 
		# and the next waypoint.
		curr_position = agent.mission.mission[agent.mission.position] 
		next_waypoint = agent.mission.mission[agent.mission.position+1]
		
		# Create two paths for the agent using Dijkstra's algorithm to the 
		# next_waypoint. The Dijkstra method will output the same class but 
		# with the path located into the path_class applied as an input to the 
	    # method. For example, if the path_class applied is "agent.paths.distance", 
		# the Dijkstra method will return the path in "agent.paths.min_dist.path".
		agent = agent.Dijkstra(curr_position, next_waypoint, agent.paths.min_dist, method="Distance")
		agent = agent.Dijkstra(curr_position, next_waypoint, agent.paths.max_prob, method="Probability")		
		
		# Since the path has yet to be validated, we should analyse both paths 
		# using PRISM and select the path which has the best validated probability 
		# of success. So... create the first action set. 
		action_1 = Prism.Generate_Action(agent.map, num_solutions=1, initial_guess=agent.paths.min_dist.path)
		action_2 = Prism.Generate_Action(agent.map, num_solutions=1, initial_guess=agent.paths.max_prob.path)

		# Run PRISM validation on the 1st path 
		code = Prism.Create_Model(agent.map, curr_position, next_waypoint, action_1[0,:])
		file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
		agent.paths.min_dist.valid = Prism.Simulate(prism_path, file_path+model_name, output_files=True)
		    
		# Run PRISM validation on the 2nd path
		code = Prism.Create_Model(agent.map, curr_position, next_waypoint, action_2[0,:])
		file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
		agent.paths.max_prob.valid = Prism.Simulate(prism_path, file_path+model_name, output_files=True)
		
		# Select the path based on validation probability as the PCTL relationship for 
		# PRISM will return the maximum probability value. Therefore...
		if agent.paths.min_dist.valid >= agent.paths.max_prob.valid:
			agent.paths.selected = deepcopy(agent.paths.min_dist)	# Deep copy the min_dist
		else:
			agent.paths.selected = deepcopy(agent.paths.max_prob)	# Deep copy the max_prob

		# Based on the path distance, compute the estimated completion time based on the agent's speed
		agent.paths.selected.time = agent.paths.selected.length / agent.dynamics.velocity

		# Create a distance vector where each entry in the vector is a cumulative distance 
		# value from the previous node. 
		agent = Simulation.Path_Cum_Dist(agent)

		print(f"\nThe agent begins task {agent.mission.position+1} and will path to node: {next_waypoint} using path {agent.paths.selected.path}")

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
	def Path_Cum_Dist(agent):
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
	