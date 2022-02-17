# -*- coding: utf-8 -*-
import heapq, random, glob, subprocess
import numpy as np
from copy import deepcopy
import numpy as np
from random import randint, uniform

# =============================================================================
# Environment Creation Interface
# =============================================================================
''' The environment creates an interface for the entity which stores information 
    relating to the entities interpretation of the environment. The graph class 
    also creates all connections and maps for the environment, and includes path 
    finding using Dijkstra's algorithm.    
'''
class Graph:
    def __init__(self, n_nodes, ID, n_probs=3):
        # Setup nodes and variables
        self.n_nodes = n_nodes  # Number of nodes in the environment
        self.n_probs = n_probs  # Number of probabilities (success, fail, return)
        self.dist_array = np.zeros(shape=(n_nodes, n_nodes))
        self.prob_array = np.zeros(shape=(n_nodes, n_nodes))
        self.map = dict()       # Default environment map
        self.heat_map = dict()  # Adjusted heatmap.
        self.ID = ID
        self.connections = None # Map connections        

        # Variables for information.
        self.path = None
        self.paths = self.__Path()
        
        # Initilise the dynamics of the agent (technically... kinematics)
        self.dynamics = self.__Dynamics()
        
        # Task Specific Variables 
        self.mission = self.__Mission()
                
    # =============================================================================
    # Create Connections and Add Connections
    # -----------------------------------------------------------------------------
    # The Create_Connection method takes in a list of connections between two nodes 
    # and adds the connection to the distance and probability arrays.
    # =============================================================================
    def Create_Connections(self, connections):
        # Create connections between the nodes
        for c in connections:
            node_1 = c[0]
            node_2 = c[1]
            distance = c[2]
            probability = c[3]
            
            # Update the value in the distance array for the first node and the 
            # second node. This should also be repeated as the array should be 
            # mirrored about the diagonal.
            self.dist_array[node_1-1, node_2-1] = distance
            self.dist_array[node_2-1, node_1-1] = distance
            
            # We may have a second input which requires use of the probability array
            # If this is the case, populate that array in the same manner as the previous
            # distance array.
            if probability is not None:
                self.prob_array[node_1-1, node_2-1] = probability
                self.prob_array[node_2-1, node_1-1] = probability

        # Apply the connections to the class
        self.connections = connections
      
    # =============================================================================
    # Create Map
    # -----------------------------------------------------------------------------
    # Create the environment map based on the probability and distance matrices 
    # which were populated using the Create_Connections method. 
    #
    # A map can also be created from a previously defined map. This helps eliminate
    # stochastic behaviour when creating the map, since many variables are randomly
    # computed, no two maps will ever be the same.
    # =============================================================================
    def Create_Map(self, env_map=None):
        # The input variable env_map allows this instance to be created based on 
        # a previously created map. However, if this value is None, we should create 
        # the map from scratch. 
        if env_map is None:
            # When creating the connections between each node, the connections are 
            # defined with two values: distance and probabilty. However, only the 
            # distance value is required with the probability optional. Therefore, 
            # we need to check to make sure we have information within the probability 
            # array in order to use that information to populate the values. Therefore, 
            # we should check to see if the probabiltiy array has been populated by 
            # checking its maximum value. If the array has values, the maximum value 
            # will be greater than zero, which was the default.
            if self.prob_array.max() > 0:
                # Create the map using information from the distance and probabilty 
                # matrices.
                for i in range(self.dist_array.shape[0]):
                    self.map[i+1] = dict()
                    for j in range(self.dist_array.shape[1]):
                        if self.dist_array[i,j] != 0:
                            prob_success = self.prob_array[i,j] # Obtain the prob of success
                            
                            # Generate random probabilities based on the probability of success 
                            # using the internal method __Random_Probabilities. This returns 
                            # three variables, a fail, a return state and a total value. The
                            # total value should always equal 1. 
                            prob_fail, prob_return, total = self.__Random_Probabilities(prob_success)
            
                            
                            # For some scenarios where we have a probability of success,
                            # the value which corresponds a failure may not have a return state, 
                            # meaning if the state fails, there is no second changes. This is 
                            # determined based on the number of probabilities assigned to the class
                            # when it is created. For example, if the number  of probs is 2,
                            # then the return probability should be zero and it's value added onto 
                            # the failure probability.
                            if self.n_probs == 2:
                                prob_fail += prob_return
                                prob_return = 0
                            
                            # Create a map strucutre based on the nodes i and j and create a 
                            # dictionary of values corresponding to the probabilities/ 
                            self.map[i+1][j+1] = {"Distance"    : self.dist_array[i,j],
                                                  "Success"     : prob_success,
                                                  "Return"      : prob_return,
                                                  "Fail"        : prob_fail,
                                                  "Total"       : total}
            
            # The probabilty array indicates  it has not been populated and therefore
            # we should create the map using only the distance array. This is mainly 
            # used for population of maps where probabilty values are not important 
            # or do not exist, such as when creating maps for humans which do not have 
            # probabilities of success defined. 
            else:
                # Create the map using information from ONLY the distance 
                for i in range(self.dist_array.shape[0]):
                    self.map[i+1] = dict()
                    for j in range(self.dist_array.shape[1]):
                        if self.dist_array[i,j] != 0:
                            self.map[i+1][j+1] = {"Distance" : self.dist_array[i,j]}
        
        # If the env_map variable is NOT None, then we should create this instance 
        # from a previously defined map.                     
        else:
            # Crate a deep copy of the map to prevent values being known based on 
            # memory addresses. 
            self.map = deepcopy(env_map)
            
            # If the map which was created does not have the same number of probabilities
            # that are equal to 3, then we need to adjust the probabilities. 
            if self.n_probs == 2:
                # We should add the "return" probability to the "fail" probability, and
                # reset the "return" probabilty to zero as it won't be used. 
                for node in self.map:
                    for conn in self.map[node]:
                        self.map[node][conn]['Fail'] = np.round(self.map[node][conn]["Fail"] + self.map[node][conn]['Return'], 2)
                        self.map[node][conn]['Return'] = 0
                        
    # =============================================================================
    # Create Random Probabilities for Environment Map
    # -----------------------------------------------------------------------------
    # This is an internal method for creating the environment map based on the dist 
    # and probability arrays. It uses the probability of success to create a randomly 
    # generated return and failure state, which when added together with the success 
    # value MUST equal one. 
    # =============================================================================
    def __Random_Probabilities(self, success):
        total = 0 # Create initial total for checking the value inside the while statement
        
        # We cannot exit the function if the total value is NOT 1.
        while total != 1.00:
            # Using the success value, determine the maximum value used to create 
            # the returna and failure states. 
            remainder = 1 - success
            
            # We need to first find the decimal place to allow a better represenation 
            # of the number of places we must determine the values for. For example, 
            # if the success if 0.4, then the fail states are comprised of 0.6, with
            # one decimal place. However, if the success is 0.95, the fail state need 
            # to be must smaaller and defined quantities within the round function, as
            # they must be comprised of only 0.05. 
            dec_place = str(success)[::-1].find('.')
            
            # Create one ranndom value using a uniform distribution and apply the 
            # decimal place derived before for accuracy. 
            val_1 = np.round(random.uniform(remainder*0.1, remainder), dec_place)
            
            # Determine the second value by subtracing the first random value from the 
            # remainder. 
            val_2 = np.round(remainder - val_1, 3)
    
            # Ideally, we want the return prob to be larger than the fail prob. Therefore
            # use a simple if statement to catch the larger number and then assign the 
            # final states values. 
            if val_1 >= val_2:
                ret = val_1
                fail = val_2
            else:
                ret = val_2 
                fail = val_1
            
            # Add the probabilities to ensure they add up to one
            total = success + fail + ret
            
        # If we break out of the loop, return the values as outputs from the 
        # function. We also return the total value as a check for later. 
        return fail, ret, total
       

    # =============================================================================
    # Create Random Missions
    # -----------------------------------------------------------------------------
    #
    # =============================================================================
    def Random_Mission(self, n_nodes, hold_rate=0.8, num_phases=3, max_unordered=100):
        min_node = min(self.map)
        max_node = max(self.map)
    
        self.dynamics.position = randint(min_node, max_node)
        self.mission.start = self.dynamics.position
        self.mission.tasks = [randint(min_node, max_node) for i in range(n_nodes)]
    
        headers = list()
        counter = 0

        # Iterate through the number of nodes to create the headers list. 
        for i in range(n_nodes-1):

            # To prevent large unordered tasks from accumulating, use 
            # the max_unordered variable to limit the number of consecutive 
            # unordered tasks.
            if counter < max_unordered:
                if uniform(0, 1) <= hold_rate:
                    headers.append("C")
                    counter += 1 # Add to the counter
                else:
                    headers.append("H")
                    counter = 0 # Reset the counter
            else:
                headers.append("H")
                counter = 0 # Reset the counter

        # Make the last index a hold statement 
        headers.append("H")

        self.mission.headers = headers

    # =============================================================================
    # Update heat map    
    # -----------------------------------------------------------------------------
    # If a path is created for one entity, this path can be used to adjust the prob
    # of success in another entities map by applying a scaling factor to nodes within 
    # the path.
    # =============================================================================
    def Update_Heat(self, path, scale=0.5):
        # Create a heat map based on the default map.
        self.heat_map = deepcopy(self.map)
        
        # Determine which connections should be adjusted based on the path.
        # Iterate through each connection...
        for c in self.connections:
            # if either the first or second cell in the connection is also in the
            # the path, the success should be adjusted
            if (c[0] in path) or (c[1] in path):
                self.heat_map[c[0]][c[1]]["Success"] *= scale
                self.heat_map[c[0]][c[1]]["Fail"] += self.heat_map[c[0]][c[1]]["Success"]

                self.heat_map[c[1]][c[0]]["Success"] *= scale
                self.heat_map[c[1]][c[0]]["Fail"] += self.heat_map[c[1]][c[0]]["Success"]                
                
    
    # =============================================================================
    # Dijkstra's Algorithm for Path Finding
    # =============================================================================
    def Dijkstra(self, start, final, path_class=None, method="Distance", secondary="Success", map=None):       
        # We want to be able to use updated heatmaps, so if the map variable is None, use the default map, 
        # else, set the map to be the map passed into the method. 
        if map is None:
            map = self.map  # Set the map to be the default map

        if method == "Distance":
            # We are using Dijkstra's algorithm to minimise distance.
            nodes = {k : np.inf for k in map.keys()}
            nodes[start] = 0 # Set the edge we are starting at to the be min value.
            prev_node = dict()
            
            # create connection list and use heapq priorty queue
            connections = list()
            heapq.heappush(connections, (0, start))
            # While we have a node in the heap
            while connections:
                # obtain the current lowest distance in the heap array
                curr_distance, curr_node = heapq.heappop(connections)
                
                # Iterate through each neighbour at the current node location
                # for neighbour, edge_dist in self.map[curr_node].items():
                for connection in map[curr_node].items():
                    # the connection variable will return a list with two values:
                    #  - the connecting neighbour
                    #  - and the values of distance, success.... etc within the map for this neighbour
                    neighbour = connection[0]
                    edge = connection[1]["Distance"]
                    
                    # Calculate the new distance using the current distance and the distance 
                    # to the neighbour
                    new_distance = curr_distance + edge 
            
                    # If the new distannce is less than the known distance to that neighbour... update its value.
                    if new_distance < nodes[neighbour]:
                        nodes[neighbour] = new_distance # Update the distance to the node
                        prev_node[neighbour] = curr_node # store the previous neighbour for backtracking the path.
                        
                        # push the connection for the current distance and neighbour
                        heapq.heappush(connections, (new_distance, neighbour))
        
        elif method == "Probability":
            # We are using Dijkstra's to maximise probability.
            nodes = {k : 0 for k in map.keys()}
            nodes[start] = 1 # Set the edge we are starting at to the maximum expected value.
            prev_node = dict()
            
            # create connection list and use heapq priorty queue
            connections = list()
            heapq.heappush(connections, (0, start))
            
            # While we have a node in the heap
            while connections: 
                # obtain the current lowest 
                curr_prob, curr_node = heapq.heappop(connections)
                
                # Iterate through each neighbour at the current node location
                # for neighbour, edge_dist in self.map[curr_node].items():
                for connection in map[curr_node].items():
                    # the connection variable will return a list with two values:
                    #  - the connecting neighbour
                    #  - and the values of distance, success.... etc within the map for this neighbour
                    neighbour = connection[0]
                    edge = connection[1]["Success"]# + connection[1]["Return"]

                    # calculate the new probability to the neighbour
                    if curr_prob == 0:
                        new_probability = edge
                    else:
                        new_probability = curr_prob * edge
    
                    # If teh new probability is less than the known probability to that neighbour, update its value
                    if new_probability > nodes[neighbour]:
                        nodes[neighbour] = new_probability
                        prev_node[neighbour] = curr_node
                        
            
                        # Push the connection for the current probability and neighbour 
                        heapq.heappush(connections, (new_probability, neighbour))
        
        # If the input method was not recognised.
        else:
            print("Optional methods are: 'Distance' or 'Probability'")
            
        
        # Create the path
        path_position = final
        path = [path_position]
        
        # To create the path we need to traverse the predecessor locations from 
        # the final position to the start location.
        while path_position is not start:
            next_position = prev_node[path_position]
            path.append(next_position)
            path_position = next_position
        path.reverse() # Since the path will be backwards, we need to reverse the path
        
        # Now we have a path, we should also return the value of the opposite method. 
        # So if we used distance, we should determine the probability of the path. 
        if method == "Distance":
            probability = 1
            for i in range(len(path)-1):
                x_1 = path[i]
                x_2 = path[i+1]
                probability *= (map[x_1][x_2]["Success"] + map[x_1][x_2]["Return"])
            distance = nodes[final]
        
        elif method == "Probability":
            distance = 0
            for i in range(len(path)-1):
                x_1 = path[i]
                x_2 = path[i+1]
                distance += map[x_1][x_2]["Distance"]
            probability = nodes[final]
            
        # The method input has a "path_class" which indicates a map has been 
        # applied for which the path solution can been appended to.
        if path_class is not None:
            # Create the exportation for the class.
            path_class.path = path
            path_class.length = distance
            path_class.prob = probability
            path_class.valid = None # Reset the validation value.
            return self
        
        # The method does not have a class to update the values in. Therefore, 
        # we will return the raw values for the path, distance and probability.
        else: 
            return path, distance, probability
    
    # =============================================================================
    #  Method for validating a created path using the PRISM class.
    # -----------------------------------------------------------------------------
    # When creating a path using Dijkstra's algorithm, the algorithm only considers 
    # the path which has the highest chance of success, immediately, and does not 
    # consider the fact that a path has a return state, allowing the agent to try
    # the edge again. For this reason, PRISM is used to validate the probability of
    # successfully reaching the end state in a systematic way. For these reasons, 
    # the probabilty of success obtained through PRISM is usually larger than that 
    # returned using Dijkstra. 
    # =============================================================================
    def Validate_Path(self, prism_path, path):
        
        # To validate the path using PRISM we need to create the appropriate 
        # actions for the PRISM model using the created path.
        action = Prism.Generate_Action(self.map, num_solutions=1, initial_guess=path)
        
        # Generate PRISM code and compile the PRISM model.
        code = Prism.Create_Model(self.map, self.position, path[-1], action[0,:])
        file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
        
        # Run the PRISM model and obtain the validation value from the PCTL.
        validation = Prism.Simulate(prism_path, file_path+model_name, output_files=True)
        
        
        return validation
    
    # =============================================================================
    # Compile Mission
    # -----------------------------------------------------------------------------
    # When the task breakdown has been created, the mission can be applied to the 
    # agent class by compiling inside this methodology. This will create the 
    # necessary information for the mission inside the agent's class.
    # =============================================================================
    def Compile_Mission(self, sub_tasks):
        # Update the mission breakdown 
        self.mission.breakdown = deepcopy(sub_tasks)

        # Set the phase information 
        self.mission.n_phase = len(sub_tasks)   # Determine number of phases 
        self.mission.i_phase = 1                # Set the current phase index to 1
        self.mission.c_phase = True             # Set the complete phase bool to False

        self.mission.i_task = 1
        self.mission.t_task = 1


    # =============================================================================
    # Sub-Class for Environment Paths
    # -----------------------------------------------------------------------------
    # This sub-class creates the path class that is applied to the agent for 
    # navigation throught the environment. It cannot be called externally by the 
    # class through methods, and is initalised during creation.
    #
    # The purpose of this class is to create a set of initialised variables for 
    # simulation, such as the selected path and the parameters (distance/prob) 
    # for the selected path.
    # =============================================================================
    class __Path: 
        def __init__(self):
            self.selected = self.__Instance()       # Selected path for simulated
            self.max_prob = self.__Instance()       # Path created using Dijkstra (probability)
            self.min_dist = self.__Instance()       # Path created using Dijkstra (distance)
            # self.history = np.empty(shape=(0,2))    # Historic path information for the agent
            self.history = list()
            
        class __Instance:
            def __init__(self):
                self.path = None            # Actual path 
                self.i_path = 0             # Current position (index) along the path
                self.n_return = 0           # Counter for return states
                self.length = None          # Distance of the path
                self.prob = None            # Probability of completing the path (Dijkstra)
                self.time = None            # Time expected to complete the path 
                self.valid = None           # Validation probability from Prism.
                self.dist_cum = None        # Iterative cummulative distance of the path.
        
    # =============================================================================
    # Sub-Class for Environment Tasks 
    # -----------------------------------------------------------------------------
    # The sub-class Task initialises the task for the agent governed by the main 
    # Mission class. 
    #
    # Each mission is comprised of a series of tasks where each task is represented 
    # by a location within the environment. Since tasks do not necessary have to be 
    # conducted in sequantial order, tasks are assigned headers which are characters 
    # that describe whether the task is an ordered or un-ordered process.
    # =============================================================================
    class __Mission: 
        def __init__(self):
            self.start = None       # Start location for the agent.
            self.tasks = None       # List of tasks as node locations
            self.phase = None       # What is the current phase task list?!
            self.n_phase = None     # Number of phases in a mission
            self.i_phase = None     # Current phase index in the mission
            self.i_task  = None     # Current task index in a specific phase
            self.t_task = None      # Total tasks completed
            self.c_phase = None     # Boolean for whether the current phase is complete
            

            self.breakdown = None   # Full mission breakdown

            self.index = 0          # Index of the sub-mission
            self.progress = None    # Progress of the task by the agent
            self.position = 0       # Position of the task? 
            self.complete = False   # List of completed tasks
            self.failed = False     # Boolean for whether the mission failed.
            self.mission = None     # When the mission order has been selected... it goes here! 
            self.time = 0           # Timer for mission progress

            # Each task is comprised of a series of locations defined by nodes 
            # within the environment. Each task is assigned a header which defines 
            # its characteristics for completion. 
            # Headers: C ("Check"), H ("Hold")
            self.headers = None
            
     
    # =============================================================================
    # Agent Dynamics
    # -----------------------------------------------------------------------------
    # This class sets the dynamics of the agent during simulation
    # =============================================================================
    class __Dynamics:
        def __init__(self):
            self.velocity = 0.5     # transitional velocity (m/s)
            self.rotation = 0.5     # angular velocity (rad/s)
            self.position = None    # Position of the agent (node position)
            self.yaw      = 0.0     # Yaw angle of the agent (rad)

            # History = [mission sub-task, sub-task, position, curr_pos, next_pos, act_pos, p_succ, p_ret, uniform]
            self.history  = np.empty(shape=(0, 9))
            
# =============================================================================
# PRISM Interface Class
# 
# Models can be checked using PRISM with code developed at runtime to support
# the PRISM interface.
# =============================================================================
class Prism:  
    
    # =============================================================================
    # Generate Actions
    # -----------------------------------------------------------------------------
    # When creating a PRISM model to check and validate paths, we do not want PRISM 
    # to perform policy synthesis, but rather be used as a simulation tool which 
    # follows preset actions. These actions correspond to movement between nodes 
    # where the movement was predetermined through the path obtained from the path 
    # finding algorith. Therefore, this path is applied into the method as the 
    # initial_guess, and is then used to create appropriate actions. 
    #
    # If no path is applied to the method, the a random set of actions are created
    # giving the model completely random movements. This can be used for meta-heuristic 
    # optimisation. 
    # =============================================================================
    def Generate_Action(nodes, num_solutions, initial_guess=None):
        # This creats a random set of actions for the PRISM model which can 
        # then allow some sort of metaheuristic optimisation to be performed 
        # to locate the optimal action set. However, if a path is applied as 
        # the initial_guess variable, then a path is applied from Dijkstra's
        # and optimisation will not actually be performed. 
        num_actions = [len(nodes[node]) for node in nodes]
        action_array = np.zeros(shape=(num_solutions, len(num_actions)), dtype=np.int32)
    
        for j, action in enumerate(action_array):
            for i, val in enumerate(action):
                max_action = num_actions[i]
                action = random.randint(1, max_action)
                action_array[j,i] = action
        
        # Because we have used Dijkstra's algorithm to create the path, yet we 
        # randomly initialised the action array for the PRISM model, we need to locate 
        # the actions which correspond to movement through the space between each node 
        # identified on the Dijkstra solution.
        if initial_guess is not None:
            for n in range(len(initial_guess)-1):
                curr_node = initial_guess[n]     # current node in the iteration
                next_node = initial_guess[n+1]   # next node we intend to move to from the curr_node
                action_array[0, curr_node-1] = list(nodes[curr_node].keys()).index(next_node) + 1
        
        return action_array
    
    # =============================================================================
    # Create PRISM Model
    # -----------------------------------------------------------------------------
    # To create PRISM models at run-time, this method is used. It is created based 
    # on the entire map of the environment (nodes) as well as the start and final 
    # location. The action array is only passed into the method as this corresponds
    # to the preset path which was used to determine appropriate actions to 
    # successfully navigate to the final state. 
    # =============================================================================
    def Create_Model(nodes, start_location, final_location, actions):
        PREAMBLE = list()           # initial code for the PRISM model
        WORKFLOW = list()           # main body of the PRISM model
        REWARD_DISTANCE = list()    # Reward structure for the PRISM model
    
        # Create the preamble
        PREAMBLE.append("// Code generaetion for preamble.\n")
        PREAMBLE.append("mdp\n\n")
    
        # model parameters
        PREAMBLE.append("// Model parameters\n")
        PREAMBLE.append(f"const int start = {start_location};\n")
        PREAMBLE.append(f"const int final = {final_location};\n")
        PREAMBLE.append("\n")
    
        # begin action synthesis
        PREAMBLE.append("// Create action selections\n")
        for i in range(len(nodes)):
            act = actions[i]
            PREAMBLE.append(f"const int a_s{i+1} = {act}; \t// Selected action in the range 1 to {len(nodes[i+1])};\n")
    
        # create WORKFLOW module 
        WORKFLOW.append("\n\n\n")
        WORKFLOW.append("module workflow\n")
        WORKFLOW.append("\tend : bool init false;\n")
        WORKFLOW.append(f"\ts : [0..{len(nodes)}] init {start_location};\n\n")
        for node in nodes:
            act = 0 # start action counter
            for trans in nodes[node]:
                act += 1
                # Each state/action in the workflow is comprised of four parts (condition, success, return, and fail)
                WORKFLOW.append(f"\t[s{node}_s{trans}] (s={node}) & (a_s{node}={act}) & (!end) -> ")  # Condition
                WORKFLOW.append(f"{nodes[node][trans]['Success']}:(s'={trans}) + ")                  # Success state
                WORKFLOW.append(f"{nodes[node][trans]['Return']}:(s'={node}) + ")                    # Return state
                WORKFLOW.append(f"{nodes[node][trans]['Fail']}:(s'=0); \n")                          # Fail state
    
                # Each of the state/action needs to also have a reward structure
                REWARD_DISTANCE.append(f'\t[s{node}_s{trans}] true : {nodes[node][trans]["Distance"]};\n')
    
        WORKFLOW.append("\n\t[end] (!end) & (s=0 | s=final) -> (end'=true);\n")
        WORKFLOW.append("\nendmodule\n\n\n")
    
        # reward structure
        REWARD_DISTANCE.insert(0, '\nrewards "distance" \n')
        REWARD_DISTANCE.append("endrewards")
               
        # Compile
        model = PREAMBLE + WORKFLOW + REWARD_DISTANCE
        
        return model 
    
    # =============================================================================
    # Export Model 
    # -----------------------------------------------------------------------------
    # The PRISM model which was created is exported using this method based on the 
    # model file, file name and path. Each entry in the model is written line by
    # line. 
    # =============================================================================
    def Export_Model(model, file_name=None, path=""):
        if file_name is None:
            n_files = len(glob.glob1(path, "*.prism")) + 1
            file_name = f"Model_{n_files}.prism"
            
        with open(path + file_name, 'w') as f:
            for row in model:
                f.write(row)
                
        return path, file_name

    # =============================================================================
    # Simulate
    # -----------------------------------------------------------------------------
    #  
    # =============================================================================
    def Simulate(prism_path, model, output_files=False):
        if output_files:
            # Output the policy and states files as well
            policy_path = model[0:-6]+".tra"
            states_path = model[0:-6]+".sta"
            expression = [f"{prism_path}", f"{model}", "-pctl", "Pmax=? [F (end & s=final)]", 
                          "-exportadv", f"{policy_path}", "-exportmodel", f"{states_path}"]

            # expression = [f"{prism_path}", f"{model}", "-pctl", "Pmin=? [F (end & s=final)]", 
            #               "-exportadv", f"{policy_path}", "-exportmodel", f"{states_path}"]
        else:
            # Don't output the policy and states, but run the model
            expression = [f"{prism_path}", f"{model}",  "-pctl", 'Pmax=? [F (end & s=final)]']
                      
        # Run the expression in the command line using subprocess
        process = subprocess.Popen(expression, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        out = out.decode('utf-8')
        out = out.split()
        
        # The output from running the PRISM model is stored within the out variable, 
        # where the result is located in the immediate cell after the cell which reads 
        # "Result:"
        result = None
        for j, entry in enumerate(out):
            if entry == "Result:":
                 result = (float(out[j+1]))
                 break
        
        if result is None: 
            print("Something went wrong. Check the model path.")
            
        return result
        










