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

# =============================================================================
# Solve Initial Path for Agent
# =============================================================================
agent_start = 25
agent_final = 17

agent_path_dist, agent_dist_dist, agent_dist_prob = agent.Dijkstra(agent_start, agent_final, path_class=None, method="Distance")
agent_path_prob, agent_prob_dist, agent_prob_prob = agent.Dijkstra(agent_start, agent_final, path_class=None, method="Probability")

print("Agent Initial")
print(agent_path_dist)
print(agent_path_prob)

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

agent_path_dist1, agent_dist_dist1, agent_dist_prob1 = agent.Dijkstra(agent_start, agent_final, path_class=None, method="Distance", map=agent.heat_map)
agent_path_prob1, agent_prob_dist1, agent_prob_prob1 = agent.Dijkstra(agent_start, agent_final, path_class=None, method="Probability", map=agent.heat_map)

print("Agent Initial")
print(agent_path_dist)
print(agent_path_prob)
