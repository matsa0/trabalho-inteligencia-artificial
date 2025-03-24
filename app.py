from agent import Agent
import numpy as np

agent = Agent()

regions_color_and_cost = {
    # intervalo de cores HSV para garantir que a cor seja capturada
    "piso_seco": ([0, 0, 180], [180, 40, 255], 1),  
    "piso_molhado": ([90, 50, 50], [130, 255, 255], 3),  
    "fiacao_exposta": ([0, 50, 50], [10, 255, 255], 6),  
    "porta": ([10, 100, 20], [30, 255, 200], 4), 
    "parede": ([0, 0, 50], [180, 20, 100], 0),  
}

group_positions = {
    "E": ((6, 40), 1),   # Eleven 
    "D": ((5, 7), 1),    # Dustin
    "L": ((20, 10), 1),  # Lucas
    "M": ((17, 37), 1),  # Mike
    "W": ((30, 11), 1),  # Will
    "X": ((41, 40), 1)   # Exit
}

image_path = "img/mapa_laboratorio.png"

matrix = agent.proccess_imagem(image_path, regions_color_and_cost, group_positions)
np.savetxt("mapa_laboratorio.txt", matrix, fmt="%d")

best_path = agent.find_best_path(matrix, group_positions)
agent.visualize_agent_moves(matrix, best_path, group_positions)