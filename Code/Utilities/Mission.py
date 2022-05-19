import numpy as np
from copy import deepcopy
from itertools import permutations


# =============================================================================
# Mission Class
# =============================================================================
# The mission class creates the solution for multiple mission profiles. This is 
# first performed by creating a secondary map solely for nodes located within 
# the mission assigned to the agent (agent.mission.tasks).
#
# A breakdown of the mission is performed to analyse discrete section of the 
# mission which could be performed in any order, with permutations created for 
# every single possible configuration of the mission. 
#
# The solution is solved using a least distance analysis whilst also validating 
# the probabiltiy of the solution using PRISM.
# =============================================================================
class Mission:
	def __init__(self, agent):
		self = self.Create_Connections(agent)
		self.environment = None	# Initialise the environment variable.
		self.complete  = False	# Check for if the mission has been completed
		self.max_steps = 100	# Maximum number of steps for the simulation

	# =============================================================================
	# Create Connection    
	# =============================================================================
	# Obtain the agent's (entity) current position and list of tasks within their 
	# mission objective and create a set of connections which will be used to create 
	# an environment map for the mission. 
	# =============================================================================
	def Create_Connections(self, agent):
		# Current location when the method was called.
		self.start = agent.dynamics.position
		
		# Create a deepcopy of the agent's mission and task list
		self.tasks = deepcopy(agent.mission.tasks)
		self.headers = deepcopy(agent.mission.headers)

		# Append the start location to the task and add a holder "S" implying "start" 
		# to the task_holders list. Only perform this if the start_node != first node
		if self.start != self.tasks[0]:
			self.tasks.insert(0, self.start)
			self.headers.insert(0, "S")
		else:
			# ensure the first holder indicates the start
			self.headers[0] = "S"      
		
		# The agent and human classes were created based on a set of connections
		# which were imported from Maps. However, we are going to create a new map using 
		# solely the mission tasks, which means we need to create a new set of connections 
		# specific to these locations. 
		self.connections = list()

		# Iterate through the list of tasks
		for i in range(len(self.tasks)):
			for j in range(len(self.tasks)):
				start = self.tasks[i] # start node
				final = self.tasks[j] # connecting node

				''' When creating a set of connections, we need the distance of the edge 
					and the probability of successfully traversing the edge. Since these 
					quantities are not known, we will use Dijkstra's algorithm to solve 
					the distance and probability for each edge connection - forming the 
					mission. 
				
					For this, we will only use the path with the best probability 
					rather than the path of least distance. 
					
					Since we are using Dijkstra's algorithm to determine the probabiltiies,
					they are going to be less than those determining systematically using
					PRISM. So they will need validation.
				'''
			
				# agent_path_dist, agent_dist_dist, agent_dist_prob = agent.Dijkstra(start, final, path_class=None, method="Distance")
				agent_path_prob, agent_prob_dist, agent_prob_prob = agent.Dijkstra(start, final, path_class=None, method="Probability")

				# Append the probabilities and distance values obtained from Dijkstra's to the 
				# mission connections.
				self.connections.append([self.tasks[i], 
										 self.tasks[j], 
										 round(agent_prob_dist, 2), 
										 round(agent_prob_prob, 6)])    
				   
		# return self just to stop it overwriting itself 
		return self
	
	
	# =============================================================================
	# Mission Breakdown   
	# -----------------------------------------------------------------------------
	# The mission breakdown method uses the mission tasks and the headers to create 
	# partitions of the mission, where each partition contains a region of tasks 
	# which are un-ordered based on the heading character. Regions which are 
	# un-ordered can be perfomred in any order and need to be determine for 
	# efficiency
	# 
	# When a mission breakdown occurs and a mission is partitioned into multiple 
	# stages, by default each stage ends with the start state of the next scenario. 
	# However, the optional variable end_state causes this, and if the boolean is 
	# set to False, each stage will not end of the start state of the next stage.
	# =============================================================================
	def Breakdown(self):
		# with each location assigned a holder value indicating what needs to be done at 
		# that location. We create a new data structure which uses the applied handle to 
		# partition the mission into a series of sub-tasks, where the sub-tasks are compiled
		# with a start and final location, and all tasks inbetween can be performed in any 
		# order. 
		sub_tasks = dict()
		n_mission = 0 # counter for the sub-mission task directory 

		# Itereate through each task in the mission and create sub-tasks based 
		# on the value of corresponding header for that specific task. 
		for idx, holder in enumerate(self.headers):
			# If this is the first task in the mission, we need to initilise the mission
			if idx == 0:
				# Create a new-sub task directory 
				sub_tasks[n_mission] = dict()

				# If the value of holder is "S", that indicates the start of the mission
				if holder == "S":
					# Initialise the start location for this sub-task
					sub_tasks[n_mission]["S"] = self.tasks[idx]

					# Create a list of permutable tasks within this sub-task which are 
					# un-ordered as defined by the "U" character.
					sub_tasks[n_mission]["U"] = list()
					sub_tasks[n_mission]["H"] = list()

				# We are at the start but don't have a "S" state, therefore
				# else:
					# sub_tasks[n_mission]["S"] = None
					# sub_tasks[n_mission]["U"] = [self.tasks[idx]]
					# sub_tasks[n_mission]["H"] = list()

			else:
				# If the value of the holder is "U", then these tasks are permutable 
				# and should be appended to the current sub-task directory inside the 
				# permutable task list.
				if holder == "U":
					sub_tasks[n_mission]["U"].append(self.tasks[idx])
				  
				# if the value of the holder is "H", then this task is allocated to the 
				# human and can be performed as an unordered task, but must be completed 
				# before the phase can end. 
				elif holder == "H":
					sub_tasks[n_mission]["H"].append(self.tasks[idx])

				# If the holder value is "O". then this indicates an ordered task, which 
				# signals the end of the currnet phase and the start location for the next 
				# phase of the mission. 
				elif holder == "O":
					# Use the end_state boolean to determine whether the end state of the 
					# mission stage should end on the start condition of the next stage.
					sub_tasks[n_mission]["E"] = self.tasks[idx]
				   
					# if the idx value is equal to the len(task)-1, that indicates that the 
					# current sub-task end is also the end of the overall mission. Therefore,
					# if idx is less than the len(task)-1, we should create a new sub-task 
					# as the mission has not been completed yet. 
					if idx < len(self.tasks)-1:
						# Begin the next sub-task by increasing the sub-count and creating a new
						# sub-directory where the start of this task is this location. 
						n_mission += 1
						sub_tasks[n_mission] = dict()
						sub_tasks[n_mission]["S"] = self.tasks[idx]
						sub_tasks[n_mission]["U"] = list()
						sub_tasks[n_mission]["H"] = list()
	                    
		return sub_tasks	   
				   
	# =============================================================================
	# Permute            
	# -----------------------------------------------------------------------------
	# The permute method takes the mission breakdown of the tasks and iterates 
	# through all of the sub-tasks. For each sub-task the permuted tasks, identified 
	# in the "C" list are then permuted for every single configuration. The mission 
	# sub-tasks are then compiled into the permuted configurations, creating every 
	# instance of the mission profile which can then be evaluated to obtain the 
	# solution. 
	# =============================================================================
	def Permute(self, sub_tasks, apply_end_state=True):
		# for each sub-task created within the mission.breakdown list, identify all 
		# permutable tasks that should be located within the "U" list. 
		for i in range(len(sub_tasks)):
			# Create a list of permutations for every permutable task.
			permute = list(permutations(sub_tasks[i]["U"]))
			permute = [list(p) for p in permute]
		   
			# Iterate through each permutation and append the 
			for p in permute:
				p.insert(0, sub_tasks[i]["S"])

				# We need to check if we have an end condition.
				if apply_end_state is True:
					p.append(sub_tasks[i]["E"])
				
			# Add the permuted task order to the breakdown variable
			sub_tasks[i]["Permuted"] = permute
			
		return sub_tasks
		
	# =============================================================================
	# Solve  
	# -----------------------------------------------------------------------------  
	# The solve method takes the sub_taks variable and performs an iterative 
	# anlaysis over each path to determine a cumulative distance and probability 
	# value for ever path permutation that creates the mission. This process is not 
	# efficient and represents a 'cowboy' solution... I am digusted with myself for 
	# this, but hey... it works! 
	# =============================================================================
	def Solve(self, sub_tasks):
		# Iterate through each set of sub-tasks in the sub-task variable.
		for i in range(len(sub_tasks)):
			# Create a solution dataset for the sub_tasks 
			sub_tasks[i]["Solutions"] = dict() 
			sub_tasks[i]["Solutions"]["Results"] = np.empty(shape=(0,2))
			
			# Iterate through the permuted paths in the sub-tasks variable and determine 
			# the distances and probabilities based on moving between each node individually 
			# from the map.
			paths = sub_tasks[i]["Permuted"] 
			for path in paths:
				dist = 0 # reset the distance value for the current path
				prob = 1 # reset the probability value for the current path
				
				# Iterate through each node that creates the path j
				for j in range(len(path)-1):
					s1 = path[j]    # current node
					s2 = path[j+1]  # next node
		
					# If s1 is NOT the same as s2, we can use these values to add onto the
					# distance metric and also multiple the probability value. If s1 is the 
					# as s2, we are not technically moving since the nodes are the same. Therefore
					# we would do nothing. 
					if s1 != s2:
						dist += self.environment.map[s1][s2]["Distance"]
						prob *= self.environment.map[s1][s2]["Success"]

				# Append the cummulative solution for the distance and probabilty to the solution 
				# array in the sub_tasks variable. 
				sub_tasks[i]["Solutions"]["Results"] = np.vstack((sub_tasks[i]["Solutions"]["Results"], np.array([dist, prob]).reshape(1,2)))

			
			# After analysing all of the paths in the sub-directory for each sub-task, compile 
			# the minimum distance paths. 
			sub_tasks[i]["Solutions"]["Distance"] = dict()

			# Determine the path value which has the minimum distance
			min_dist_value = sub_tasks[i]["Solutions"]["Results"][:,0].min()
			sub_tasks[i]["Solutions"]["Distance"]["Min Value"] = min_dist_value

			# Determine the index values of the path which has the minimum distance
			min_dist_index = [i_1 for i_1, x in enumerate(sub_tasks[i]["Solutions"]["Results"][:,0]) if x == sub_tasks[i]["Solutions"]["Distance"]["Min Value"]]
			sub_tasks[i]["Solutions"]["Distance"]["Min Index"] = min_dist_index

			# Iteratively locate paths which have the minimum distance
			min_dist_paths = [sub_tasks[i]["Permuted"][i_2] for i_2 in sub_tasks[i]["Solutions"]["Distance"]["Min Index"]]
			sub_tasks[i]["Solutions"]["Distance"]["Paths"] = min_dist_paths
			
			# After analysing all of the paths in the sub-directory for this mission, compile the 
			# maximum probability paths
			sub_tasks[i]["Solutions"]["Probability"] = dict()

			# Determine the path value which has the highest probabilty
			max_prob_value = sub_tasks[i]["Solutions"]["Results"][:,1].max()
			sub_tasks[i]["Solutions"]["Probability"]["Max Value"] = max_prob_value

			# Determine the index values of the path which has the highest probability
			max_prob_index = [i_1 for i_1, x in enumerate(sub_tasks[i]["Solutions"]["Results"][:,1]) if x == sub_tasks[i]["Solutions"]["Probability"]["Max Value"]]
			sub_tasks[i]["Solutions"]["Probability"]["Max Index"] = max_prob_index

			# Iteratively locate paths which have the highest probability.
			max_prob_paths = [sub_tasks[i]["Permuted"][i_2] for i_2 in sub_tasks[i]["Solutions"]["Probability"]["Max Index"]]
			sub_tasks[i]["Solutions"]["Probability"]["Paths"] = max_prob_paths
			
			
		return sub_tasks
		
	   
	   
	   
# =============================================================================
# Preset mission classes
# ============================================================================= 
class Preset_Missions:
	def Mission_One(start, final):
		# The first mission will simulate an environment monitoring procedure
		# where the robot moves between rooms in the environment and checks to 
		# to sure that no issues or dangers are present. 
		#
		# During this mission, the human is free to roam the environment and 
		# will not be allocated any missions to perform. 
		tasks = [start, 3, 6, 9, 12, 16, 24, 28, 29, final]
		
		headers = ['O', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'O']  

		return tasks, headers 

	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
   