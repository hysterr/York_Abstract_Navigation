#%% Importation
from Environment import Graph, Prism
from Maps import Risk, Bungalow

#%% Create Environemnt
# Create connections for the environment
risk_matrix = Risk()
connections = Bungalow(risk_matrix)

# Create environment for the agent
num_nodes = max(max(connections))
agent = Graph(n_nodes=num_nodes, n_probs=3) 
agent.Create_Connections(connections)
agent.Create_Map()

#%% Create Task
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