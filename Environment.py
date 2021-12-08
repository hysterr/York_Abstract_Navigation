# -*- coding: utf-8 -*-
import heapq, random, glob, subprocess
import numpy as np
from copy import deepcopy
import numpy as np

# =============================================================================
# Environment Creation Interface
# =============================================================================
class Graph:
    def __init__(self, n_nodes, n_probs=3):
        # Setup nodes and variables
        self.n_nodes = n_nodes
        self.n_probs = n_probs
        self.dist_array = np.zeros(shape=(n_nodes, n_nodes))
        self.prob_array = np.zeros(shape=(n_nodes, n_nodes))
        self.map = dict() 
        self.heat_map = dict()
        
        # Variables for information.
        self.position = None
        self.path = None
        self.paths = self.__Path()
        self.speed = 0.5
        
        # Task Specific Variables 
        self.task = self.__Task()
        
    # =============================================================================
    # Sub-Class for Environment Paths
    # =============================================================================
    class __Path: 
        def __init__(self):
            self.selected = self.__Instance()
            self.probability = self.__Instance()
            self.distance = self.__Instance()
            self.history = np.empty(shape=(0,2))
            
        class __Instance:
            def __init__(self):
                self.path = None # Actual path 
                self.length = None # Distance of the path
                self.prob = None # Probability of completing the path (Dijkstra)
                self.time = None # Time expected to complete the path 
                self.valid = None # Validation probability from Prism.
                self.dist_cum = None # Iterative cummulative distance of the path.
                self.progress_dist = None # Progress variable for simulation. 
                self.progress_time = None # Progress variable for time along the path.
        
    # =============================================================================
    # Sub-Class for Environment Tasks 
    # =============================================================================
    class __Task: 
        def __init__(self):
            self.task = None
            self.progress = None
            self.position = 0
            self.complete = list()
        
    # =============================================================================
    # Create Connections and Add Connections 
    # =============================================================================
    def Create_Connections(self, connections):
        # Create connections between the nodes
        for c in connections:
            self.Add_Connection(c[0], c[1], c[2], c[3])
            
    def Add_Connection(self, start, final, distance, probability=None):
        self.dist_array[start-1, final-1] = distance
        self.dist_array[final-1, start-1] = distance
        
        # We may have a second input, which requires use of the probability array
        # If this is the case, populate that array in the same manner as the previous
        # distance array.
        if probability is not None:
            self.prob_array[start-1, final-1] = probability
            self.prob_array[final-1, start-1] = probability
            
        
    # =============================================================================
    # Create map
    # - Create the environment map based on the probability and distance matrices
    # - a previously defined map can be used to create a second instance. 
    #   - This prevents random generation from impacting an environment.     
    # =============================================================================
    def Create_Map(self, env_map=None):
        # The input variable env_map allows this instance to be created based on 
        # a previously created map. However, if this value is None, we should create 
        # the map from scratch. 
        if env_map is None:
            # check to see if we have actual values in the probability array. This 
            # will occur if the max value in the array is not zero. 
            if self.prob_array.max() > 0:
                # Create the map using information from the distance and probabilty 
                # matrices.
                for i in range(self.dist_array.shape[0]):
                    self.map[i+1] = dict()
                    for j in range(self.dist_array.shape[1]):
                        if self.dist_array[i,j] != 0:
                            prob_success = self.prob_array[i,j]
                            prob_fail, prob_return, total = self.__Random_Probabilities(prob_success)
                            
                            # Depending on the number of probabilities within the class,
                            # the values should change. For example, if the number is 2,
                            # then the return probability should be zero and added onto the
                            # failure probability.
                            if self.n_probs == 2:
                                prob_fail += prob_return
                                prob_return = 0
                            
                            self.map[i+1][j+1] = {"Distance" : self.dist_array[i,j],
                                                  "Success" : prob_success,
                                                  "Return" : prob_return,
                                                  "Fail" : prob_fail,
                                                  "Total" : total}
    
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
            # Crate a deep copy of the map.
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
    # Update heat map    
    # - Paths can be input into the class to produce a scalable map of intensity,
    #   allowing the environment to have an adjusted heat map for probabilities.
    # =============================================================================
    def Update_Heat(self, connections, path, scale=0.5):
        # Create a heat map based on the default map.
        self.heat_map = deepcopy(self.map)
        
        # Determine which connections should be adjusted based on the path.
        # Iterate through each connection...
        for c in connections:
            # if either the first or second cell in the connection is also in the
            # the path, the success should be adjusted
            if (c[0] in path) or (c[1] in path):
                self.heat_map[c[0]][c[1]]["Success"] *= scale
                self.heat_map[c[0]][c[1]]["Return"] += self.heat_map[c[0]][c[1]]["Success"]
                
    
    # =============================================================================
    # Create Random Probabilities for Environment Map
    # =============================================================================
    def __Random_Probabilities(self, success):
        total = 0
        
        while total != 1.00:
            remainder = 1 - success
            dec_place = str(success)[::-1].find('.')
            val_1  = np.round(random.uniform(remainder*0.1, remainder), dec_place)
            val_2 = np.round(remainder - val_1, 3)
    
            if val_1 >= val_2:
                ret = val_1
                fail = val_2
            else:
                ret = val_2 
                fail = val_1
            
            # Add the probabilities to ensure they add up to one
            total = success + fail + ret
        return fail, ret, total
           

    # =============================================================================
    # Dijkstra's Algorithm for Path Finding
    # =============================================================================
    def Dijkstra(self, start, final, path_class=None, method="Distance", secondary="Success"):       
        if method == "Distance":
            # We are using Dijkstra's algorithm to minimise distance.
            nodes = {k : np.inf for k in self.map.keys()}
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
                for connection in self.map[curr_node].items():
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
            nodes = {k : 0 for k in self.map.keys()}
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
                for connection in self.map[curr_node].items():
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
                probability *= (self.map[x_1][x_2]["Success"] + self.map[x_1][x_2]["Return"])
            distance = nodes[final]
        
        elif method == "Probability":
            distance = 0
            for i in range(len(path)-1):
                x_1 = path[i]
                x_2 = path[i+1]
                distance += self.map[x_1][x_2]["Distance"]
            probability = nodes[final]
            
        if path_class is not None:
            # Create the exportation for the class.
            path_class.path = path
            path_class.length = distance
            path_class.prob = probability
            path_class.valid = None # Reset the validation value.
            
        
            return self
        
        else: 
            return path
    
    # =============================================================================
    #  Method for validating a created path using the PRISM class.
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
    
    def TSP_Cowboy(self, connections, task):
        # For each node location that creates the task, we need to evaluate movement between 
        # the nodes. This is first achieved by iterating thorugh the list to create connections, 
        # and from the connections, using Dijkstra's algorithm to obtain the distances and 
        # probabilities.
        t_conns = list()
        for i in range(len(task)):
            for j in range(i+1, len(task)):
                start = task[i] # first node
                final = task[j] # second node
                
                # Obtain path from Dijkstra's
                agent_path_dist, agent_dist_dist, agent_dist_prob = self.agent.Dijkstra(start, final, method="Distance")
                agent_path_prob, agent_prob_dist, agent_prob_prob = self.agent.Dijkstra(start, final, method="Probability")
                
                # Append values to the t_conn list.
                t_conns.append([task[i], task[j], round(agent_prob_dist, 2), round(agent_prob_prob, 6)])
    
    
# =============================================================================
# PRISM Interface Class
# =============================================================================
class Prism:    
    def Generate_Action(nodes, num_solutions, initial_guess=None):
        num_actions = [len(nodes[node]) for node in nodes]
        action_array = np.zeros(shape=(num_solutions, len(num_actions)), dtype=np.int32)
    
        for j, action in enumerate(action_array):
            for i, val in enumerate(action):
                max_action = num_actions[i]
                action = random.randint(1, max_action)
                action_array[j,i] = action
        
        # Because we have used Dijkstra's algorithm to create the geodesic path, yet we 
        # randomly initialised the action array for the PRISM model, we need to locate 
        # the actions which correspond to movement through the space between each node 
        # identified on the Dijkstra solution.
        if initial_guess is not None:
            for n in range(len(initial_guess)-1):
                curr_node = initial_guess[n]     # current node in the iteration
                next_node = initial_guess[n+1]   # next node we intend to move to from the curr_node
                action_array[0, curr_node-1] = list(nodes[curr_node].keys()).index(next_node) + 1
        
        return action_array
    
    def Create_Model(nodes, start_location, final_location, actions):
        PREAMBLE = list()
        WORKFLOW = list()
        REWARD_DISTANCE = list()
    
        # Create the preamble
        PREAMBLE.append("// Code generaetion for preamble.\n")
        PREAMBLE.append("mdp\n\n")
    
        # model parameters
        PREAMBLE.append("// Model parameters\n")
        PREAMBLE.append(f"const int start = {start_location};\n")
        PREAMBLE.append(f"const int final = {final_location}; \n")
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
    
    
    def Export_Model(model, file_name=None, path=""):
        if file_name is None:
            n_files = len(glob.glob1(path, "*.prism")) + 1
            file_name = f"Model_{n_files}.prism"
            
        with open(path + file_name, 'w') as f:
            for row in model:
                f.write(row)
        return path, file_name

    def Simulate(prism_path, model, output_files=False):
        if output_files:
            # Output the policy and states files as well
            policy_path = model[0:-6]+".tra"
            states_path = model[0:-6]+".sta"
            expression = [f"{prism_path}", f"{model}", "-pctl", "Pmax=? [F (end & s=final)]", 
                          "-exportadv", f"{policy_path}", "-exportmodel", f"{states_path}"]
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
        
        
        
        

# =============================================================================
# Create entity function allows the creation of environments using an iterative
# function based on a connection list.
# =============================================================================
# def Create_Entity(connections, n_probs):
#     n_connections = max(max(connections))
#     env = Graph(n_connections, n_probs)    
    
#     # Create connections between the nodes
#     for c in connections:
#         env.add_connection(c[0], c[1], c[2], c[3])

#     return env

# # =============================================================================
# # After a path has been produced for the human, the connections for the agent 
# # should be adjusted to produce heated environment, based on the scale function
# # =============================================================================
# def Update_Heatmap(connections, path, scale=0.5):
#     new_connections = deepcopy(connections)
#     for c in new_connections:
#         # if a node goes to a connection which is on the path of the human, increase the risk.
#         if (c[0] in path) or (c[1] in path):
#             c[3] *= 0.5 # increase the risk by reducing the chance of success

#     return new_connections












