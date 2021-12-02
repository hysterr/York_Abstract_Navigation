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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
  
        
    
    