# =============================================================================
# Import
# =============================================================================
from Environment import Graph, Prism
from Maps import Risk, Bungalow
from copy import deepcopy 
from itertools import permutations 
import numpy as np 

# =============================================================================
# Create Environment
# =============================================================================
risk_matrix = Risk()
connections = Bungalow(risk_matrix)

num_nodes = max(max(connections))
agent = Graph(n_nodes=num_nodes, n_probs=3)
agent.Create_Connections(connections)
agent.Create_Map()

# Create environment for the human
human = Graph(n_nodes=num_nodes, n_probs=2)
human.Create_Connections(connections)
human.Create_Map(agent.map)

# We will use PRISM to validate paths.
PRISM_PATH = '/Users/jordanhamilton/Documents/PRISM/bin/prism'

#%% ===========================================================================
# Solve Initial Path for Agent
# =============================================================================
agent_start = 25
agent_final = 17

agent_path_dist, agent_dist_dist, agent_dist_prob = agent.Dijkstra(agent_start, agent_final, path_class=None, method="Distance")
agent_path_prob, agent_prob_dist, agent_prob_prob = agent.Dijkstra(agent_start, agent_final, path_class=None, method="Probability")

# validate the path for the agent 
action_1 = Prism.Generate_Action(agent.map, 1, agent_path_dist) 
code = Prism.Create_Model(agent.map, agent_start, agent_final, action_1[0,:])
file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
valid_1 = Prism.Simulate(PRISM_PATH, file_path+model_name, output_files=True)

action_2 = Prism.Generate_Action(agent.map, 1, agent_path_prob) 
code = Prism.Create_Model(agent.map, agent_start, agent_final, action_2[0,:])
file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
valid_2 = Prism.Simulate(PRISM_PATH, file_path+model_name, output_files=True)

print("Agent Initial")
print(agent_path_dist, round(valid_1,4))
print(agent_path_prob, round(valid_2,4))


# =============================================================================
# Solve Initial Path for Human
# =============================================================================
human_start = 15
human_final = 6

human_path_dist, human_dist_dist, human_dist_prob = human.Dijkstra(human_start, human_final, path_class=None, method="Distance")
human_path_prob, human_prob_dist, human_prob_prob = human.Dijkstra(human_start, human_final, path_class=None, method="Probability")

print("Human Initial")
print(human_path_dist)
print(human_path_prob)

# =============================================================================
# Solve Path for Robot when in Conflict with Human
# =============================================================================
agent.Update_Heat(connections, path=human_path_dist, scale=0.5)

# Validate path for agent 
action_3 = Prism.Generate_Action(agent.heat_map, 1, agent_path_prob) 
code = Prism.Create_Model(agent.heat_map, agent_start, agent_final, action_3[0,:])
file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
valid_3 = Prism.Simulate(PRISM_PATH, file_path+model_name, output_files=True)
print("Agent old path is now: ", round(valid_3, 4))

agent_path_dist1, agent_dist_dist1, agent_dist_prob1 = agent.Dijkstra(agent_start, agent_final, path_class=None, method="Distance", map=agent.heat_map)
agent_path_prob1, agent_prob_dist1, agent_prob_prob1 = agent.Dijkstra(agent_start, agent_final, path_class=None, method="Probability", map=agent.heat_map)

# Validate path for agent
action_4 = Prism.Generate_Action(agent.heat_map, 1, agent_path_prob1) 
code = Prism.Create_Model(agent.heat_map, agent_start, agent_final, action_4[0,:])
file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
valid_4 = Prism.Simulate(PRISM_PATH, file_path+model_name, output_files=True)

print("Agent Updated")
print(agent_path_prob1, round(valid_4, 4))
