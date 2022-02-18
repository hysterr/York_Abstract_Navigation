# =============================================================================
	# Select Path
	# -----------------------------------------------------------------------------
	# The method "Select_Path" identifies the agent's location relative to the task 
	# and mission and creates a path to the next waypoint. 
	#
	# Paths are stored within the agent's path class (agent.paths) and are selected 
	# based on a PRISM validation analysis. 
	# =============================================================================
	def Select_Path(agent, prism_path=None, validate=True, human=None):
		# We will use PRISM to validate paths.
		if prism_path is None:
			prism_path = '/Users/jordanhamilton/Documents/PRISM/bin/prism'

		# If the argument 'human' is None, this indicates a solo path finding venture
		if human is None:
			# We need to first create a path for the agent between the current location 
			# and the next waypoint.
			curr_position = agent.dynamics.position
			next_waypoint = agent.mission.phase[agent.mission.i_task]

			# Create two paths for the agent using Dijkstra's algorithm to the 
			# next_waypoint. The Dijkstra method will output the same class but 
			# with the path located into the path_class applied as an input to the 
		    # method. For example, if the path_class applied is "agent.paths.distance", 
			# the Dijkstra method will return the path in "agent.paths.min_dist.path".
			agent = agent.Dijkstra(curr_position, next_waypoint, agent.paths.min_dist, method="Distance")
			agent = agent.Dijkstra(curr_position, next_waypoint, agent.paths.max_prob, method="Probability")		

			# We want to perform a certain action only if the ID of the class indicates 
			# we are currently working on the agent.
			if agent.ID == "Agent":
				# Check to see if the paths should be validated using PRISM (This adds time for simulation).
				if validate:
					agent = Simulation.__Validate(agent, prism_path)

				# If the validate boolean is False, select the path found to have the highest probability 
				# of success
				elif not validate:
					agent.paths.selected = deepcopy(agent.paths.max_prob)

			# If we are selecing a path for the human, we will just take the least distance path
			elif agent.ID == "Human":
				agent.paths.selected = deepcopy(agent.paths.min_dist)

			# Based on the path distance, compute the estimated completion time based on the agent's speed
			agent.paths.selected.time = agent.paths.selected.length / agent.dynamics.velocity

			# Create a distance vector where each entry in the vector is a cumulative distance 
			# value from the previous node. 
			agent = Simulation.Path_Cummulative_Distance(agent)

			# If the curr_position == next_waypoint, the agent will remain in the same location
			# but we still want to evaluate the path. 
			if curr_position == next_waypoint:
				agent.paths.selected.path.append(agent.paths.selected.path[0])

			print(f"The agent begins task {agent.mission.t_task} and will path from node {curr_position} to node: {next_waypoint} using path {agent.paths.selected.path}")

			return agent

		# If a human class has been passed into the method, we will need to perform 
		# path planning for the agent with consideration to the predictive movement 
		# of the human. 
		elif human is not None:
			
			# If the human has a task in the current phase, the length of the phase will 
			# be greater than 0. if this is true, use the phase to path for the human
			if len(human.mission.phase) > 0:
				# First obtain the path of least disance for the human 
				human_pos = human.dynamics.position
				human_way = human.mission.phase[human.mission.i_task]
				
			# If this is not true, the human does not have a task to perform, and we will 
			# assume for the purposes of planning, the agent stays in the same location
			else:
				human_pos = human.dynamics.position
				human_way = human_pos

			human = human.Dijkstra(human_pos, human_way, human.paths.min_dist, method="Distance")
			human.paths.selected = deepcopy(human.paths.min_dist)

			# Second, update the agent's environment view to factor the human's path
			agent.Update_Heat(path=human.paths.selected.path, scale=0.75)

			# Third, perform path finding for the agent
			agent_pos = agent.dynamics.position
			agent_way = agent.mission.phase[agent.mission.i_task]

			agent = agent.Dijkstra(agent_pos, agent_way, agent.paths.min_dist, method="Distance",    map=agent.heat_map)
			agent = agent.Dijkstra(agent_pos, agent_way, agent.paths.max_prob, method="Probability", map=agent.heat_map)		

			# Check to see if the paths should be validated using PRISM (This adds time for simulation).
			if validate:
				agent = Simulation.__Validate(agent, prism_path)

			# If the validate boolean is False, select the path found to have the highest probability 
			# of success
			elif not validate:
				agent.paths.selected = deepcopy(agent.paths.max_prob)

			# Based on the path distance, compute the estimated completion time based on the agent's speed
			agent.paths.selected.time = agent.paths.selected.length / agent.dynamics.velocity

			# Create a distance vector where each entry in the vector is a cumulative distance 
			# value from the previous node. 
			agent = Simulation.Path_Cummulative_Distance(agent)

			# If the curr_position == next_waypoint, the agent will remain in the same location
			# but we still want to evaluate the path. 
			if agent_pos == agent_way:
				agent.paths.selected.path.append(agent.paths.selected.path[0])


			# Have a differnet prompt for when the human stays in the smae location as opposed to when performing a task
			if human_pos == human_way: 
				print(f"The human has no allocated task.")
			else:
				print(f"The human begins task {human.mission.t_task+1} and will path from node {human_pos} to node: {human_way} using path {human.paths.selected.path}")

			print(f"The agent begins task {agent.mission.t_task} and will path from node {agent_pos} to node: {agent_way} using path {agent.paths.selected.path}")

			return agent, human