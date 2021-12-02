# -*- coding: utf-8 -*-
import heapq, random
import numpy as np
from copy import deepcopy

# =============================================================================
# Environment Creation Interface
# =============================================================================
class Graph:
    def __init__(self, n_nodes, n_probs=3):
        self.n_nodes = n_nodes
        self.n_probs = n_probs
        self.dist_array = np.zeros(shape=(n_nodes, n_nodes))
        self.prob_array = np.zeros(shape=(n_nodes, n_nodes))
        self.map = dict() 
        self.heat_map = dict()
         
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
           

    def Dijkstra(self, start, final, method="Distance", secondary="Success"):       
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
            
        return path, distance, probability
    
    
def TSP(self, connections, task):
    pass

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












