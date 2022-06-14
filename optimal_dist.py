import numpy as np

distance_matrix = np.loadtxt('input.txt')

route = np.loadtxt('optimal.txt')

total_dist = 0
for i in range(28):
    total_dist += distance_matrix[int(route[i]) - 1, int(route[i + 1]) - 1]
    
total_dist += distance_matrix[int(route[28]) - 1, int(route[0]) - 1]
    
print(total_dist)