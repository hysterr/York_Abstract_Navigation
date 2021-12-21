#%% Importation
from Environment import Graph, Prism
from Maps import Risk, Bungalow, LivingArea
from copy import deepcopy
from itertools import permutations
import numpy as np

#%% ===========================================================================
# Create Environment Objects
# =============================================================================
# Create connections for the environment
risk_matrix = Risk()
connections = LivingArea(risk_matrix)

# The environment is created as two parts, one for the agent and one for 
# the human. The human is created with only two probabilities and uses 
# the map created by the agent as part of its creation policy to ensure 
# consistency within the environment.
# Create environment for the agent
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
# Mission Definement
# =============================================================================
agent.position = 1 # current position of the robot (node)
human.position = 18 # current position of the human (node)

robot_pos_ind  = 0  # current index of the robot along the path.
robot_task_ind = -1 # current index of the robot progress for the overall mission. (-1 indicates start location.)

human.speed = 0.3 # speed in m/s for human
agent.speed = 0.1 # speed in m/s for robot

# Task:   1. Check cupboard at 1 (locate item)
#                1.1 - Request human retrieves item and takes to work surface (4)
#         2. Check fridge at 9 (locate item)
#                2.1 - Request human retrieves item and takes to work surface (5)
#         3. Move to table at 10 (check its clear)
#         4. Check cupboard at 3 (locate item)
#                4.1 - Request human retrieves item and takes to work surface (4)
#         5. Move to table at 10 (Hold...)
agent.task.task = [1, 9, 10, 3, 10]
agent.task.position = 0 # Set the index of the agent's task to 0. 
agent.task.progress = [agent.task.task[agent.task.position]]

# Each location along the mission/task will have an intermediate task for the robot to perform
# such as "check" or "hold". The status of each intermediate task is described using "C" and "H"
# and these holders will be used to request the human perform some action when the robot reaches 
# one of the these states.
# mission_task_holders = ["C", "C", "C", "C", "H"]
mission_task_holders = ['C', 'C', 'C', 'H', 'C', 'C', 'H', 'C', 'C', 'C', 'C', 'C', 'H']

#%% ===========================================================================
# Travelling Salesman Solution for the Mission
# =============================================================================
# We have been given a task located in agent.task.task, but this task order does 
# not necessarily have to be conducted in sequential order. The ordering of the 
# task should be determined based on the mission_task_holder, where if the task 
# has a holder value of "C", that implies it is just a "check" function, where 
# checks can be conducted in any order. The value "H" implies a "hold" funtion,
# where the agent must complete the prior tasks before holding in this location
# until further instruction. 
start_node = agent.position

# Create new copies of the task and task_holders
task = deepcopy(agent.task.task)
task_holders = deepcopy(mission_task_holders)

# Append the start location to the task and add a holder "S" implying "start" 
# to the task_holders list.
task.insert(0, start_node)
task_holders.insert(0, "S")

# --- TEST TASKS ----
task = [1, 5, 8, 10, 1, 2, 8, 15, 8, 2, 1, 2, 4, 5]
task_holders = ['S', 'C', 'C', 'C', 'H', 'C', 'C', 'H', 'C', 'C', 'C', 'C', 'C', 'H']

# Previously, the agent and human classes were created based on a set of connections
# which were imported from Maps. However, we are going to create a new map using 
# solely the mission tasks, which means we need to create a new set of connections 
# specific to these locations. 
mission_connections = list()

# Iterate through the list of tasks
for i in range(len(task)):
    for j in range(len(task)):
        start = task[i] # start node
        final = task[j] # connecting node

        # Use Dijkstra's to a path between the start and final nodes based on the 
        # agent's map. The values for distance and probability will then be used to 
        # create the connections, forming the map for the mission.
        agent_path_dist, agent_dist_dist, agent_dist_prob = agent.Dijkstra(start, final, path_class=None, method="Distance")
        agent_path_prob, agent_prob_dist, agent_prob_prob = agent.Dijkstra(start, final, path_class=None, method="Probability")
        
        # Append the probabilities and distance values obtained from Dijkstra's to the 
        # mission connections.
        mission_connections.append([task[i], task[j], round(agent_prob_dist, 2), round(agent_prob_prob, 6)])

# Create a class for the mission so the problem can be solved using the connection 
# just created. 
Mission = Graph(n_nodes=num_nodes, n_probs=3)
Mission.Create_Connections(mission_connections)
Mission.Create_Map()

''' Now the map has been created for connections existing solely within the mission,
    the least distance between the nodes, or travelling salesman problem, can be 
    solved. (..Note.. This is a COWBOY solution -- very dirty solution) '''

# The mission has been instigated as a series of locations that should be visited
# with each location assigned a holder value indicating what needs to be done at 
# that location. We create a new data structure which uses the applied handle to 
# partition the mission into a series of sub-tasks, where the sub-tasks are compiled
# with a start and final location, and all tasks inbetween can be performed in any 
# order. 
task_breakdown = dict()
subs = 0 # Counter for sub-mission task directory
for idx, holder in enumerate(task_holders):
    # If the holder value is "S", that indicates the start 
    # of the mission. 
    if holder == "S":
        task_breakdown[subs] = dict() # create a sub-directory
        task_breakdown[subs]["S"] = task[idx] # initialise the start location
        task_breakdown[subs]["C"] = list() # create a list for permutable tasks

    # If the holder is "C", then these tasks are permutable, and should be appended
    # to the current mission's sub-directory inside the permutable task list.
    elif holder == "C":
        task_breakdown[subs]["C"].append(task[idx])

    # If the holder is "H", then this indicates a hold function, which signals 
    # the end of the current sub-task.
    elif holder == "H":
        task_breakdown[subs]["E"] = task[idx]

        # if the idx value is equal to the len(task)-1, that indicates that the 
        # current sub-task end is also the end of the overall mission. Therefore,
        # if idx is less than the len(task)-1, we should create a new sub-task 
        # as the mission has not been completed yet. 
        if idx < len(task)-1:
            # Begin the next sub-task by increasing the sub-count and creating a new
            # sub-directory where the start of this task is this location. 
            subs += 1
            task_breakdown[subs] = dict()
            task_breakdown[subs]["S"] = task[idx]
            task_breakdown[subs]["C"] = list()

# For each sub-task that has been created, create combintations of all the permutable 
# tasks. These are the tasks assigned handles of "C" within the sub-task directories.
for i in range(len(task_breakdown)):
    permute = list(permutations(task_breakdown[i]["C"]))
    permute = [list(p) for p in permute]

    # We need to append the start and final conditions for the sub-task to each of the 
    # permutations. 
    for p in permute:
        p.insert(0, task_breakdown[i]["S"])
        p.append(task_breakdown[i]["E"])

    # Add the permuted array to the task_breakdown
    task_breakdown[i]["Permuted"] = deepcopy(permute)

    # Iterate through the newly permuted paths to determine the distances and 
    # probabilities based on moving between each node individually from the map.
    task_breakdown[i]["Solutions"] = np.empty(shape=(0,2))
    for path in permute:
        dist = 0 # Reset the distance value for this path
        prob = 1 # Reset the probability value for this path

        # Iterate through each node that creates the path.
        for j in range(len(path)-1):
            s1 = path[j]    # current node
            s2 = path[j+1]  # next node 

            # If s1 is not the same as s2, we can use these value to add onto the 
            # distance metric and also multiply the probabilty value.
            if s1 != s2:
                dist += Mission.map[s1][s2]['Distance']
                prob *= Mission.map[s1][s2]['Success']

            # If s1 is the same as s2, this indicates we are moving from one node 
            # to the same node. This isn't a movement, and therefore the distance 
            # should be set to 0 and the probability should be guaranteed. 
            else:
                dist += 0
                prob *= 1

        # Append the cummulative solution for the distance and probability to the solution
        # array in the task_breakdown structure.                
        task_breakdown[i]["Solutions"] = np.vstack((task_breakdown[i]["Solutions"], np.array([dist,prob]).reshape(1,2)))
    
    # After analysing all of the paths in the sub-directory for this mission, compile the 
    # minimum distance paths
    task_breakdown[i]["Distance"] = dict()
    task_breakdown[i]["Distance"]["Min Value"] = task_breakdown[i]["Solutions"][:,0].min()
    task_breakdown[i]["Distance"]["Min Index"] = [i_1 for i_1, x in enumerate(task_breakdown[i]["Solutions"][:,0]) if x == task_breakdown[i]["Distance"]["Min Value"]]
    task_breakdown[i]["Distance"]["Paths"] = [task_breakdown[i]["Permuted"][i_2] for i_2 in task_breakdown[i]["Distance"]["Min Index"]]

    # After analysing all of the paths in the sub-directory for this mission, compile the 
    # maximum probability paths
    task_breakdown[i]["Probability"] = dict()
    task_breakdown[i]["Probability"]["Max Value"] = task_breakdown[i]["Solutions"][:,1].max()
    task_breakdown[i]["Probability"]["Max Index"] = [i_1 for i_1, x in enumerate(task_breakdown[i]["Solutions"][:,1]) if x == task_breakdown[i]["Probability"]["Max Value"]]
    task_breakdown[i]["Probability"]["Paths"] = [task_breakdown[i]["Permuted"][i_2] for i_2 in task_breakdown[i]["Probability"]["Max Index"]]

#%% ===========================================================================
# Simulation
# =============================================================================
# Mission specific checks
complete = False # Set mission complete bool to false

agent.paths.selected.time = 0 # Set the agent's path time to zero
human.paths.selected.time = 0 # Set the human's path time to zero
dt = 1e-1  # Set the time-step for the simulation

# If the robot has no path, this indicates that we are currently on the first
# task of the mission. Therefore, we should create a path without concerning 
# the human, since the human is not currently involved in the mission until 
# the robot instigates a request.
if agent.paths.selected.path is None: 
    # We need to first create a path for the robot between the current location 
    # and the next waypoint
    next_waypoint = agent.task.task[agent.task.position] # Find the next waypoint for the 

    # Create two paths for the agent using Dijkstra's algorothm to the next waypoint. The Dijkstra method 
    # will output the same class but with the path located into the path_class applied as an input to the 
    # method. For example, if the path_class applied is "agent.paths.distance", the Dijkstra method will 
    # return the path in "agent.paths.distance.path".
    agent = agent.Dijkstra(agent.position, next_waypoint, agent.paths.distance, method="Distance")
    agent = agent.Dijkstra(agent.position, next_waypoint, agent.paths.probability, method="Probability")

    # Since this is the first path, we should select the path which has the best
    # validation probability based on PRISM. So first create the PRISM action set...    
    action_1 = Prism.Generate_Action(agent.map, num_solutions=1, initial_guess=agent.paths.distance.path)
    action_2 = Prism.Generate_Action(agent.map, num_solutions=1, initial_guess=agent.paths.probability.path)
    
    # Run PRISM validation on the 1st path 
    code = Prism.Create_Model(agent.map, agent.position, next_waypoint, action_1[0,:])
    file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
    agent.paths.distance.valid = Prism.Simulate(PRISM_PATH, file_path+model_name, output_files=True)
    
    # Run PRISM validation on the 2nd path
    code = Prism.Create_Model(agent.map, agent.position, next_waypoint, action_2[0,:])
    file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
    agent.paths.probability.valid = Prism.Simulate(PRISM_PATH, file_path+model_name, output_files=True)
    
    # We should select the path based on validation probability, as the PCTL relationship for 
    # PRISM will return the maximum probability. Therefore...
    if agent.paths.distance.valid >= agent.paths.probability.valid:
        agent.paths.selected = deepcopy(agent.paths.distance)
    else:
        agent.paths.selected = deepcopy(agent.paths.probability)
        
    # Based on the path distance, compute the estimated completion time based on the agent's speed
    agent.paths.selected.time = agent.paths.selected.length / agent.speed
    
    # Create a cumulative distance vector for the path so we can keep track of the 
    # robot's progress along the path.
    agent.paths.selected.dist_cum = list() # Reset the cummulative distance of the selected path.
    cum_dist = 0
    for node in range(1, len(agent.paths.selected.path)):
        x1 = agent.paths.selected.path[node-1] # previous node
        x2 = agent.paths.selected.path[node]   # current node 
        cum_dist += agent.map[x1][x2]["Distance"]
        agent.paths.selected.dist_cum.append(cum_dist) # append the cumulative distance

# Run simulation
simulation_time = 0
while not complete:        
    # If we reach this point... we should have a path! (..Fingers crossed..)
    # Begin moving along the path... updating in time!
    agent.paths.selected.progress_dist = 0 # Reset the progress variable. 
    agent.paths.selected.progress_time = 0 # Reset the progress variable.

    # If the agent has a path then the path should lead to the next waypoint. 
    # Therefore, we can start to simulate movement towards the waypoint is the
    # robot's position does not equal the next_waypoint.
    while agent.position is not next_waypoint:
        # Update the simulation time and robot's position
        agent.paths.selected.progress_time += dt
        agent.paths.selected.progress_dist += (agent.speed * dt)
        simulation_time += dt
        
        # As the agent moves along the path, we are logging the distance it has covered
        # which can then be used to determine it's progress with regards to the mission.
        # Using the cumulative distance matriex, calculate where along the current path 
        # the agent is, and store this value as the last visited node.
        lst = [dist - agent.paths.selected.progress_dist for dist in agent.paths.selected.dist_cum]

        # The values in the list array will be negative if the agent has passed a point
        # as the agent.paths.selected.dist value will be > the cummulative distance node. 
        # Therefore, we should store the index of values that are less than 0 as this 
        # implies the agent has already passed this point
        idx = [i for i, x in enumerate(lst) if x < 0]
        
        # If the index has values, the agent has covered enough distance to pass some 
        # nodes. Therefore the position of the agent is the index + 1.
        if idx:
            new_position = agent.paths.selected.path[idx[-1]+1]

            # If the new position created this time-step is a new position when compared to the 
            # previous time-step, i.e. the agent transitions to a new node... we will update this
            # as a new position in the agent's historic information.
            if new_position != agent.position:
                agent.paths.history = np.vstack((agent.paths.history, np.array([simulation_time, new_position])))

            agent.position = new_position

        # If the index IS EMPTY, the agent has not covered enough distance to pass any of 
        # the nodes along the path. This means the most recent node was the start node. 
        else:
            agent.position = agent.paths.selected.path[0]

    # Agent has reached the waypoint, and therefore the task should be completed and moved onto 
    # the next task waypoint in agent.task.task.  
    agent.task.position += 1
    agent.task.complete.append(agent.task.progress)

    # compile next_waypoint 
    if agent.task.position == len(agent.task.task):
        complete = True
        print("Agent has reached the final position", agent.position, ". Mission is complete.")
        continue
    else:
        next_waypoint = agent.task.task[agent.task.position]

    path_string = str(agent.paths.selected.path).strip('[').strip(']')
    print("After %3.2f seconds the agent moved to position %2.0f using path: %10s (Elapsed: %3.2f seconds). \tAgent will now move to: %2.0f." % (simulation_time, agent.position, path_string, agent.paths.selected.time, next_waypoint))


    # Create the new path for the next waypoint. This creation should be done within one-time step
    agent = agent.Dijkstra(agent.position, next_waypoint, agent.paths.distance, method="Distance")
    agent = agent.Dijkstra(agent.position, next_waypoint, agent.paths.probability, method="Probability")

    # We need to somehow select which path to take.... let's just use the validation probabilty path for now. 
    agent.paths.distance.valid = agent.Validate_Path(PRISM_PATH, agent.paths.distance.path)
    agent.paths.probability.valid = agent.Validate_Path(PRISM_PATH, agent.paths.probability.path)

    # From validation, select the best path.
    if agent.paths.distance.valid >= agent.paths.probability.valid:
        agent.paths.selected = deepcopy(agent.paths.distance)
    else:
        agent.paths.selected = deepcopy(agent.paths.probability)

    # Based on the path distance, compute the estimated completion time based on the agent's speed
    agent.paths.selected.time = agent.paths.selected.length / agent.speed
    
    # Create a cumulative distance vector for the path so we can keep track of the 
    # robot's progress along the path.
    agent.paths.selected.dist_cum = list() # Reset the cummulative distance of the selected path.
    cum_dist = 0
    for node in range(1, len(agent.paths.selected.path)):
        x1 = agent.paths.selected.path[node-1] # previous node
        x2 = agent.paths.selected.path[node]   # current node 
        cum_dist += agent.map[x1][x2]["Distance"]
        agent.paths.selected.dist_cum.append(cum_dist) # append the cumulative distance

    
    # # Perform next task...
    # if mission_task_holders[robot_task_ind] == "C":
    #     # The robot is at a check mission point. this means the robot will request the human 
    #     # moves to this specific location in order to retrieve the object the robot has just
    #     # found. it is assumed the human will move along the shortest distance path.
    #     hum_waypoint = robot_position
    #     human_path_dist, human_dist_dist, human_dist_prob = human.Dijkstra(human_position, hum_waypoint)
    #     human_path_time = human_dist_dist / human_speed
    #     human_path_cdist = list()
    #     cum_dist = 0
    #     for node in range(1, len(human_path_dist)):
    #         x1 = human_path_dist[node-1] # previous node
    #         x2 = human_path_dist[node]   # current node 
    #         cum_dist += human.map[x1][x2]["Distance"]
    #         human_path_cdist.append(cum_dist) # append the cumulative distance
            
    #     break
    



















































