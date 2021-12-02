# from Graph import Create_Entity
from Graph import Graph
from Optimiser import Prism

#%% Connections 
risk_matrix = {
    "L"  : 0.95,
    "ML" : 0.90,
    "M"  : 0.87,
    "MH" : 0.85,
    "HL" : 0.80,
    "HM" : 0.75,
    "H"  : 0.65,
    "HH" : 0.50,
    "VH" : 0.40,
}

# The connections for the environment compiled as a list of lists. Each list 
# within a list contains four peices of information: 
#   1. starting node
#   2. connecting node
#   3. Linear distance 
#   4. Risk
connections = [
        [1, 2, 0.7,  risk_matrix["L"]],
        [1, 4, 1.2,  risk_matrix["ML"]],
        [1, 8, 2.2,  risk_matrix["L"]],
        [2, 3, 0.8,  risk_matrix["HL"]],
        [2, 4, 1.2,  risk_matrix["L"]],
        [2, 8, 2.8,  risk_matrix["ML"]],
        [4, 5, 0.7,  risk_matrix["HM"]],
        [4, 6, 0.8,  risk_matrix["HM"]],
        [4, 7, 0.7,  risk_matrix["HL"]],
        [4, 8, 1.5,  risk_matrix["ML"]],
        [5, 6, 0.3,  risk_matrix["HL"]],
        [5, 7, 0.4,  risk_matrix["HL"]],
        [6, 7, 0.3,  risk_matrix["HL"]],
        [8, 9, 0.5,  risk_matrix["MH"]],
        [8, 10, 0.8, risk_matrix["HL"]],
        [8, 12, 1.4, risk_matrix["HM"]],
        [9, 10, 0.7, risk_matrix["HL"]],
        [9, 11, 1.3, risk_matrix["HM"]],
        [9, 12, 1.3, risk_matrix["HM"]],
        [9, 23, 1.1, risk_matrix["HM"]],
        [9, 25, 1.2, risk_matrix["HL"]],
        [9, 26, 1.2, risk_matrix["MH"]],
        [10, 11, 1.2, risk_matrix["HL"]],
        [10, 12, 0.8, risk_matrix["HH"]],
        [10, 23, 0.5, risk_matrix["HH"]],
        [10, 25, 0.8, risk_matrix["HM"]],
        [10, 26, 1.4, risk_matrix["HL"]],
        [11, 12, 1.0, risk_matrix["HM"]],
        [11, 15, 1.7, risk_matrix["MH"]],
        [12, 13, 0.6, risk_matrix["HM"]],
        [12, 19, 0.5, risk_matrix["HM"]],
        [12, 20, 0.4, risk_matrix["VH"]],
        [13, 18, 0.2, risk_matrix["MH"]],
        [13, 19, 0.5, risk_matrix["HL"]],
        [14, 15, 0.5, risk_matrix["MH"]],
        [14, 16, 0.6, risk_matrix["ML"]],
        [14, 17, 0.7, risk_matrix["ML"]],
        [14, 18, 0.9, risk_matrix["ML"]],
        [15, 16, 0.7, risk_matrix["ML"]],
        [16, 17, 1.0, risk_matrix["M"]],
        [16, 18, 0.8, risk_matrix["ML"]],
        [17, 18, 0.5, risk_matrix["M"]],
        [18, 19, 0.4, risk_matrix["MH"]],
        [19, 21, 1.0, risk_matrix["HM"]],
        [20, 21, 0.7, risk_matrix["VH"]],
        [20, 23, 0.7, risk_matrix["HH"]],
        [21, 22, 0.7, risk_matrix["HM"]],
        [22, 24, 1.0, risk_matrix["HL"]],
        [24, 25, 1.2, risk_matrix["ML"]],
        [25, 26, 1.4, risk_matrix["ML"]],
        [26, 27, 0.4, risk_matrix["MH"]],
        [26, 28, 0.8, risk_matrix["MH"]],
        [26, 29, 0.6, risk_matrix["ML"]],
        [26, 30, 0.7, risk_matrix["MH"]],
        [27, 28, 0.3, risk_matrix["ML"]],
        [29, 30, 0.4, risk_matrix["ML"]],
    ]

#%% Create default environment for the human and the robot 
num_nodes = max(max(connections))

agent = Graph(n_nodes=num_nodes, n_probs=3) 
agent.Create_Connections(connections)
agent.Create_Map()

human = Graph(n_nodes=num_nodes, n_probs=2)
human.Create_Connections(connections)
human.Create_Map(agent.map)

#%% Solve path planning for human using Dijkstra's
start_human, final_human = (30, 6)
human_path_dist, human_dist_dist, human_dist_prob = human.Dijkstra(start_human, final_human, method="Distance")
human_path_prob, human_prob_dist, human_prob_prob = human.Dijkstra(start_human, final_human, method="Probability")

print(human_path_dist)
print(human_path_prob)

#%% Solve Agent's Path
# Using the path produced for the human, we need to adjust the heated map for the agent, 
# so this can create a better representation of the safety of the environment
# agent.Update_Heat(connections, human_path_dist, scale=0.75)

# Solve path planning for human using Dijkstra's
start_agent, final_agent = (16, 10)
agent_path_dist, agent_dist_dist, agent_dist_prob = agent.Dijkstra(start_agent, final_agent, method="Distance")
agent_path_prob, agent_prob_dist, agent_prob_prob = agent.Dijkstra(start_agent, final_agent, method="Probability")

print(agent_path_dist, agent_dist_prob)
print(agent_path_prob, agent_prob_prob)

# Model Checking
# Create and simulate Prism Model
prism_path = '/Users/jordanhamilton/Documents/PRISM/bin/prism'
action = Prism.Generate_Action(agent.map, 1, initial_guess=agent_path_prob)
code = Prism.Create_Model(agent.map, start_agent, final_agent, action[0,:])
file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
output = Prism.Simulate(prism_path, file_path+model_name, output_files=True)
print("PRISM Path Validation is: ", output)

#%% Solve Task
start_node = 22 
task = [start_node, 16, 10, 7, 3, 27, 29]

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
        agent_path_dist, agent_dist_dist, agent_dist_prob = agent.Dijkstra(start, final, method="Distance")
        agent_path_prob, agent_prob_dist, agent_prob_prob = agent.Dijkstra(start, final, method="Probability")
        
        # Append values to the t_conn list.
        t_conns.append([task[i], task[j], round(agent_prob_dist, 2), round(agent_prob_prob, 6)])
    

mission = Graph(n_nodes=num_nodes, n_probs=3)
mission.Create_Connections(t_conns)
mission.Create_Map()

# mission_path, mission_dist, mission_prob = mission.Dijkstra(22, 16, method='Probability')

# Find the least distance path
import itertools
task = task[1:]
perm = list(itertools.permutations(task))

path_dists = list()
path_probs = list()

for path in perm:
    dist = 0
    prob = 1
    
    for i in range(len(path)-1):
        if i == 0:
            s1 = start_node
            s2 = path[i]
        else:
            s1 = path[i]
            s2 = path[i+1]
            
        dist += mission.map[s1][s2]['Distance']
        prob *= mission.map[s1][s2]['Success']
        
    path_dists.append(dist)
    path_probs.append(prob)    
    
dist_ind = [i for i, x in enumerate(path_dists) if x == min(path_dists)]
prob_ind = [i for i, x in enumerate(path_probs) if x == max(path_probs)]

for ind in prob_ind:
    # Create path
    l_dist_path = list(perm[ind])
    l_dist_path.insert(0, start_node)
    print("Creating least distance path...")
    print(l_dist_path)
    print("Creating local solution...")
    for i in range(len(l_dist_path)-1):
        s1 = l_dist_path[i]
        s2 = l_dist_path[i+1]
        path, dist_1, _ = agent.Dijkstra(s1, s2, method="Probability")
        print(path, dist_1)
    


    








