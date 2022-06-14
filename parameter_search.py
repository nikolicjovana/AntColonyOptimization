import numpy as np
import pandas as pd


def iter_data(path):
    with open(path) as data_file:
        for line in data_file:
            yield from line.split()

arr = np.fromiter(iter_data('input_gr17.txt'), int)

ab = np.zeros((17, 17))
abi, abj = 0, 0
for i in range(len(arr)):
    ab[abi, abj] = arr[i]
    if arr[i] == 0:
        abj = 0
        abi += 1
    else:
        abj += 1

distance_matrix = np.loadtxt('input.txt')
#distance_matrix = ab + ab.T - np.diag(np.diag(ab))

num_nodes = distance_matrix.shape[0]
num_ants = num_nodes

visibility = 1 / distance_matrix
np.fill_diagonal(visibility, 0)

"""
Variranje rho
"""

alpha = 1
beta = 1

results = np.zeros((5, 5))

for row, rho in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
    for col, iterations in enumerate([100, 200, 300, 400, 500]):
        best_costs = []
        for simulation in range(0, 10):
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
                        #numerator = (pheromones[current_node, :] ** alpha) * (visibility[current_node, :] ** beta)  
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
                
                pheromones = (1 - rho)*pheromones
                
                for i in range(num_ants):
                    for j in range(num_nodes - 1):
                        dt = 1/distances[i]
                        pheromones[int(optimal_routes[i,j])-1,int(optimal_routes[i,j+1])-1] = pheromones[int(optimal_routes[i,j])-1,int(optimal_routes[i,j+1])-1] + dt
                
                it+=1
            
            distance = 0
            
            for j in range(num_nodes):
                
                distance += distance_matrix[int(best_route[j]) - 1, int(best_route[j + 1]) - 1]
                
            #print(distance)
            print("alpha =", alpha, "beta =", beta, "rho =", rho, "iterations =", iterations, "simulation no. =", simulation, "best distance =", distance)
            best_costs.append(distance)
        results[row, col] = np.average(best_costs)
        
df = pd.DataFrame(results)
df.columns=[0.1, 0.2, 0.3, 0.4, 0.5]
df.index=[100, 200, 300, 400, 500]
df.to_csv('results/rho.csv')

"""
Variranje alpha
"""

results = np.zeros((5, 5))
beta = 1 
rho = 0.1

for row, alpha in enumerate([0.5, 1, 2, 3, 5]):
    for col, iterations in enumerate([100, 200, 300, 400, 500]):
        best_costs = []
        for simulation in range(0, 10):
            pheromones = np.ones((num_ants, num_nodes))
            
            routes = np.zeros((num_ants, num_nodes + 1))
            rc = np.random.randint(1, num_nodes + 1, size=(num_ants, 1)).flatten()
            it = 0
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
                        #numerator = (pheromones[current_node, :] ** alpha) * (visibility[current_node, :] ** beta)  
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
                
                pheromones = (1 - rho)*pheromones
                
                for i in range(num_ants):
                    for j in range(num_nodes - 1):
                        dt = 1/distances[i]
                        pheromones[int(optimal_routes[i,j])-1,int(optimal_routes[i,j+1])-1] = pheromones[int(optimal_routes[i,j])-1,int(optimal_routes[i,j+1])-1] + dt
                
                it+=1
            
            distance = 0
            
            for j in range(num_nodes):
                
                distance += distance_matrix[int(best_route[j]) - 1, int(best_route[j + 1]) - 1]
                
            #print(distance)
            print("alpha =", alpha, "beta =", beta, "rho =", rho, "iterations =", iterations, "simulation no. =", simulation, "best distance =", distance)
            best_costs.append(distance)
        results[row, col] = np.average(best_costs)
        
df = pd.DataFrame(results)
df.columns=[0.5, 1, 2, 3, 5]
df.index=[100, 200, 300, 400, 500]
df.to_csv('results/alpha.csv')

"""
Variranje beta
"""

results = np.zeros((5, 5))
alpha = 1 
rho = 0.1

for row, beta in enumerate([0.5, 1, 2, 3, 5]):
    for col, iterations in enumerate([100, 200, 300, 400, 500]):
        best_costs = []
        for simulation in range(0, 10):
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
                        #numerator = (pheromones[current_node, :] ** alpha) * (visibility[current_node, :] ** beta)  
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
                
                pheromones = (1 - rho)*pheromones
                
                for i in range(num_ants):
                    for j in range(num_nodes - 1):
                        dt = 1/distances[i]
                        pheromones[int(optimal_routes[i,j])-1,int(optimal_routes[i,j+1])-1] = pheromones[int(optimal_routes[i,j])-1,int(optimal_routes[i,j+1])-1] + dt
                
                it+=1
            
            distance = 0
            
            for j in range(num_nodes):
                
                distance += distance_matrix[int(best_route[j]) - 1, int(best_route[j + 1]) - 1]
                
            #print(distance)
            print("alpha =", alpha, "beta =", beta, "rho =", rho, "iterations =", iterations, "simulation no. =", simulation, "best distance =", distance)
            best_costs.append(distance)
        results[row, col] = np.average(best_costs)
        
df = pd.DataFrame(results)
df.columns=[0.5, 1, 2, 3, 5]
df.index=[100, 200, 300, 400, 500]
df.to_csv('results/beta.csv')