#%% Importation
from Environment import Graph, Prism
from Maps import Risk, Bungalow, LivingArea

#%% Create Environment Objects
# Create connections for the environment
risk_matrix = Risk()
connections = LivingArea(risk_matrix)


''' The environment is created as two parts, one for the agent and one for 
    the human. The human is created with only two probabilities and uses 
    the map created by the agent as part of its creation policy to ensure 
    consistency within the environment. 
'''
# Create environment for the agent
num_nodes = max(max(connections))
agent = Graph(n_nodes=num_nodes, n_probs=3) 
agent.Create_Connections(connections)
agent.Create_Map()

# Create environment for the human
human = Graph(n_nodes=num_nodes, n_probs=2)
human.Create_Connections(connections)
human.Create_Map(agent.map)

''' We will use PRISM to validate paths. '''
PRISM_PATH = '/Users/jordanhamilton/Documents/PRISM/bin/prism'


#%% Time analysis (Discrete Robot Event Timekeeping)
robot_position = 11 # current position of the robot (node)
human_position = 18 # current position of the human (node)

robot_pos_ind  = 0  # current index of the robot along the path.
robot_task_ind = -1 # current index of the robot progress for the overall mission. (-1 indicates start location.)

# Set robot and human speeds for the time analysis 
human_speed = 0.4
robot_speed = 0.5

# Task:   1. Check cupboard at 1 (locate item)
#                1.1 - Request human retrieves item and takes to work surface (4)
#         2. Check fridge at 9 (locate item)
#                2.1 - Request human retrieves item and takes to work surface (5)
#         3. Move to table at 10 (check its clear)
#         4. Check cupboard at 3 (locate item)
#                4.1 - Request human retrieves item and takes to work surface (4)
#         5. Move to table at 10 (Hold...)
task = [1, 9, 10, 3, 10]

# Each location along the mission/task will have an intermediate task for the robot to perform
# such as "check" or "hold". The status of each intermediate task is described using "C" and "H"
# and these holders will be used to request the human perform some action when the robot reaches 
# one of the these states.
mission_task_holders = ["C", "C", "H", "C", "H"]


# Mission specific checks
complete = False
robot_path_path = None
human_path = None

task_position = 0

robot_path_time = 0
human_path_time = 0
dt = 1e-1

#%% 
while not complete:
    # If the robot has no path, this indicates that we are currently on the first
    # task of the mission. Therefore, we should create a path without concerning 
    # the human, since the human is not currently involved in the mission until 
    # the robot instigates a request.
    if robot_path_path is None:
        # create a path for the robot  
        next_waypoint = task[task_position]
        agent_path_dist, agent_dist_dist, agent_dist_prob = agent.Dijkstra(robot_position, next_waypoint)
        agent_path_prob, agent_prob_dist, agent_prob_prob = agent.Dijkstra(robot_position, next_waypoint)
        
        # Since this is the first path, we should select the path which has the best
        # validation probability based on PRISM.
        action_1 = Prism.Generate_Action(agent.map, num_solutions=1, initial_guess=agent_path_dist)
        action_2 = Prism.Generate_Action(agent.map, num_solutions=1, initial_guess=agent_path_dist)
        
        # Run the first prism validation
        code = Prism.Create_Model(agent.map, robot_position, next_waypoint, action_1[0,:])
        file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
        valid_1 = Prism.Simulate(PRISM_PATH, file_path+model_name, output_files=True)
        
        # Run the first prism validation
        code = Prism.Create_Model(agent.map, robot_position, next_waypoint, action_2[0,:])
        file_path, model_name = Prism.Export_Model(code, file_name="Model_1.prism")
        valid_2 = Prism.Simulate(PRISM_PATH, file_path+model_name, output_files=True)
        
        # Select the first path based on the validation from PRISM.
        if valid_1 >= valid_2:
            robot_path_path = agent_path_dist
            robot_path_dist = agent_dist_dist
        else:
            robot_path_path = agent_path_prob
            robot_path_dist = agent_dist_dist
        
        agent_path_time = robot_path_dist / robot_speed # compute the time to traverse the full path
        
        # Create a cumulative distance vector for the path so we can keep track of the 
        # robot's progress along the path.
        agent_path_cdist = list()
        cum_dist = 0
        for node in range(1, len(robot_path_path)):
            x1 = robot_path_path[node-1] # previous node
            x2 = robot_path_path[node]   # current node 
            cum_dist += agent.map[x1][x2]["Distance"]
            agent_path_cdist.append(cum_dist) # append the cumulative distance
            
         
    else:
        # Begin moving along the path... updating in time!
        robot_path_dist = 0
        while robot_position is not next_waypoint:
            # Updat the simulation time and robot's position/
            robot_path_time += dt
            robot_path_dist += robot_speed * dt
            
            # Based on the distance the robot has travelled, use the cummulative distance
            # matrix to calculate where the robot is currently along the path and what node
            # the agent last visited.
            lst = [dist - robot_path_dist for dist in agent_path_cdist]
            idx = [i for i, x in enumerate(lst) if x > 0]
            
            # If the index is NOT empty, the agent has not finished the path.
            if idx:
                robot_pos_ind = idx[0]
                robot_position = robot_path_path[robot_pos_ind]
            # If the index IS empty, the agent has reached the final point along path
            # so we will update the position of the robot to be the end point.
            else:
                robot_position = robot_path_path[-1]
                
        robot_task_ind += 1
        
        # Perform next task...
        if mission_task_holders[robot_task_ind] == "C":
            # The robot is at a check mission point. this means the robot will request the human 
            # moves to this specific location in order to retrieve the object the robot has just
            # found. it is assumed the human will move along the shortest distance path.
            hum_waypoint = robot_position
            human_path_dist, human_dist_dist, human_dist_prob = human.Dijkstra(human_position, hum_waypoint)
            human_path_time = human_dist_dist / human_speed
            human_path_cdist = list()
            cum_dist = 0
            for node in range(1, len(human_path_dist)):
                x1 = human_path_dist[node-1] # previous node
                x2 = human_path_dist[node]   # current node 
                cum_dist += human.map[x1][x2]["Distance"]
                human_path_cdist.append(cum_dist) # append the cumulative distance
            
        break
    


