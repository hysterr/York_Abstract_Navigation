	# =============================================================================
	# Step
	# -----------------------------------------------------------------------------
	# Perform a single discrete step	
	# =============================================================================
	def Step(agent, human=None):
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


		if human is None:
			map = agent.map # Set the map to be the default map
			human_string = ""

		else:
			map = agent.heat_map # Set the map to be the heat map

			if len(human.paths.selected.path) - 1 > human.paths.selected.i_path:
				# Step forwards for the human as well 
				human.paths.selected.i_path += 1
				human.dynamics.position = human.paths.selected.path[human.paths.selected.i_path]
				human_string = f" -- The human moved from node {human.paths.selected.path[human.paths.selected.i_path-1]} to node {human.paths.selected.path[human.paths.selected.i_path]}"

			else: 
				# The human isn't following a path... 
				human_string = f" -- The human stayed at node {human.dynamics.position}"

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
		unif = uniform(0, 1) 
		if unif <= p_success:
			# The agent successfully moves to the next node
			agent.paths.selected.i_path += 1 		# Update the position counter along the path
			agent.dynamics.position = next_node		# Update the agent's dynamic position 
			agent.paths.selected.n_return = 0		# Reset the return counter

			# Print update to console
			print(f"\tThe agent moved from node {curr_node} to {next_node} (Success)" + human_string)
		
		elif unif <= (p_success + p_return): 
			# The agent fails to move to the next node and returns to the original node
			agent.paths.selected.n_return += 1

			# if the counter indicates a return on five consecutive attempts end mission. 
			if agent.paths.selected.n_return == 5:
				# agent.mission.complete = True
				agent.mission.failed = True

			# Print update to console.
			print(f"\tThe agent returned to node {curr_node} -- counter {agent.paths.selected.n_return} (Return)" + human_string)

		else:
			# The agent suffers a failure and the mission should end.
			print(f"\tThe agent suffered a catatrophic failure when moving from node {curr_node} to {next_node}	(Fail) " + human_string)
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