import numpy as np
np.random.seed(3838144980)

distance_matrix = np.loadtxt('input.txt')

iterations = 300
num_nodes = distance_matrix.shape[0]
num_ants = num_nodes
alpha = 3
beta = 5
rho = 0.5
visibility = 1 / distance_matrix
np.fill_diagonal(visibility, 0)

pheromones = np.ones((num_ants, num_nodes))

routes = np.zeros((num_ants, num_nodes + 1))

it = 0
rc = np.random.randint(1, num_nodes + 1, size=(num_ants, 1)).flatten()
while it < iterations:
    routes[:, 0] = rc
    routes[:, num_nodes] = rc
    #svaki mrav
    for i in range(num_ants):
        
        temp_visibility = np.array(visibility)
        
        #svaki node u putanji
        for j in range(num_nodes - 1):

            cum_prob = np.zeros(num_nodes)
            
            current_node = int(routes[i, j] - 1)
            temp_visibility[:, current_node] = 0
            
            numerator = (pheromones[current_node, :] ** alpha) * (temp_visibility[current_node, :] ** beta) 
            denominator = np.sum(numerator)   
            probs = numerator / denominator
            
            cum_prob = np.cumsum(probs)
            
            r = np.random.random_sample()
            
            city = np.nonzero(cum_prob > r)[0][0] + 1
            
            routes[i, j + 1] = city
 
        last_city = list(set([i for i in range(1, num_nodes + 1)]) - set(routes[i, :-2]))[0]

        routes[i, -2] = last_city
        
    optimal_routes = np.array(routes)
    
    distances = np.zeros((num_ants, 1))
    
    for i in range(num_ants):
        
        distance = 0
        
        for j in range(num_nodes):
            
            distance += distance_matrix[int(optimal_routes[i, j]) - 1, int(optimal_routes[i, j + 1]) - 1]
            
        distances[i] = distance
        
    best_cost_index = np.argmin(distances)
    best_cost = distances[best_cost_index]   
    best_route = routes[best_cost_index, :]
    
    pheromones = (1 - rho) * pheromones
    
    for i in range(num_ants):
        for j in range(num_nodes - 1):
            dt = 1 / distances[i]
            pheromones[int(optimal_routes[i, j]) - 1,int(optimal_routes[i, j + 1]) - 1] = pheromones[int(optimal_routes[i,j])-1,int(optimal_routes[i,j+1])-1] + dt
    
    it += 1
    
distance = 0

for j in range(num_nodes):
    
    distance += distance_matrix[int(best_route[j]) - 1, int(best_route[j + 1]) - 1]
    
print(distance)