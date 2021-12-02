import glob, random, time
import numpy as np
import subprocess

# =============================================================================
# Bat Algorithm Optimisation
# =============================================================================
class Bat_Algorithm:
    def __init__(self, env, size, initial_guess=None):
        self.Population_Size = size
        self.Initial    = initial_guess
        self.Population = self.__Generate_Population(env)
        self.Fitness    = np.ones(shape=(self.Population_Size, 1), dtype=np.float32)
        self.Loudness   = np.ones(shape=(self.Population_Size, 1), dtype=np.float32)*0.75
        self.Pulse_Rate = np.zeros(shape=(self.Population_Size, 1), dtype=np.float32)
        self.Frequency  = np.zeros(shape=(self.Population_Size, self.Population.shape[1]), dtype=np.int32)
        self.Velocity   = np.zeros(shape=(self.Population_Size, self.Population.shape[1]), dtype=np.float32)

    def __Generate_Population(self, env):
        num_actions = [len(env[node]) for node in env]
        action_array = np.zeros(shape=(self.Population_Size, len(num_actions)), dtype=np.int32)
    
        for j, action in enumerate(action_array):
            for i, val in enumerate(action):
                max_action = num_actions[i]
                action = random.randint(1, max_action)
                action_array[j,i] = action
        
        # Because we have used Dijkstra's algorithm to create the geodesic path, yet we 
        # randomly initialised the action array for the PRISM model, we need to locate 
        # the actions which correspond to movement through the space between each node 
        # identified on the Dijkstra solution.
        if self.Initial is not None:
            for n in range(len(self.Initial)-1):
                curr_node = self.Initial[n]     # current node in the iteration
                next_node = self.Initial[n+1]   # next node we intend to move to from the curr_node
                action_array[0, curr_node-1] = list(env[curr_node].keys()).index(next_node) + 1
        
        return action_array
    
    def Initialise(self):
        self.Max_Movements    = 30
        self.Act_Movements    = 0
        self.Frequency_Range  = np.array([0, 1], dtype=np.float32)
        self.Loudness_Decay   = 0.5
        self.Loudness_Limit   = 0.01
        self.Pulse_Rate_Decay = 0.5
        self.Gamma            = 0.5
        self.Fitness_Stop     = 1e-3
        
        # Optimisation results arrays 
        self.Best_Position = np.empty(shape=(1, self.Population.shape[1]), dtype=np.float32)
        self.Best_Fitness = np.inf
        self.Best_Bat = 0
        
        # Iterate through initial population
        for bat in range(self.Population_Size, PRISM_ENTRY):
            self.Fitness[bat,:] = PRISM_ENTRY(env, self.Population[bat,:], FOLDER)
            
            # Locate the fittest individual
            Best_Bat = Fitness.argmin()
            Best_Fitness = Fitness[Best_Bat,:]
            Best_Position = Population[Fitness.argmin(), :]


        
        
        
        
        
        
        
        
        
        
        
        
  
        
    
    