import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

regions_color_and_cost = {
    # intervalo de cores HSV para garantir que a cor seja capturada
    "piso_seco": ([0, 0, 180], [180, 40, 255], 1),  
    "piso_molhado": ([90, 50, 50], [130, 255, 255], 3),  
    "fiacao_exposta": ([0, 50, 50], [10, 255, 255], 6),  
    "porta": ([10, 100, 20], [30, 255, 200], 4), 
    "parede": ([0, 0, 50], [180, 20, 100], 0),  
}

group_positions = {
    "E": ((6, 40), -1),   # Eleven 
    "D": ((5, 7), -2),    # Dustin
    "L": ((20, 10), -3),  # Lucas
    "M": ((18, 37), -4),  # Mike
    "W": ((30, 11), -5)    # Will
}

def proccess_imagem(image_path, regions=regions_color_and_cost, group_positions=group_positions):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (42, 42))

    # conversão BGR -> HSV para melhor detecção de cores na lib opencv
    image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    #cv2.imshow('map_hsv', image_hsv)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    map_matrix = np.zeros((42, 42), dtype=int)

    # Processar a imagem para identificar os tipos de terreno
    for terrain, (lower, upper, cost) in regions.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        # máscara booleana para cada cor
        mask = cv2.inRange(image_hsv, lower, upper)
        # pixels ficam entre 0 e 255, a posição que tem 255, recebe o custo associado
        map_matrix[mask > 0] = cost

    # Adicionar os personagens (nesse caso, o valor assciado é um identificador, não o custo)
    for name, (position, cost) in group_positions.items():
        map_matrix[position] = cost

    return map_matrix

def manhattan_distance(origin, destiny):
    # distância em linha reta entre dois pontos
    # (x1, y1) e (x2, y2)
    # |x1 - x2| + |y1 - y2|
    return abs(origin[0] - destiny[0]) + abs(origin[1] - destiny[1])


def print_manhattan_distance_rescue(group_positions=group_positions):
    start = (6, 40)

    for name, (position, _) in group_positions.items():
        if name == "E":
            pass

        distance = manhattan_distance(start, position)
        print(f"Distância Manhattan de E até a {name}: {distance}")


def A_star_search(start, goal, matrix):
    # Inicialização 
    #heapq(heap queue) - fila de prioridade
    heapq_list = []
    #nó inicial na fila de prioridade
    heapq.heappush(heapq_list, (0, start))
    G_path_cost = {start: 0}
    # A função de avaliação inicialmente é definida como a distância de manhattan do nó inicial
    F_total_cost = {start: manhattan_distance(start, goal)}
    node_visited = set() #garantir valores únicos

    print("\033[1mg(n) inicial: ", G_path_cost)
    print("\033[1mf(n) inicial: ", F_total_cost)
    print("\033[1mFila de prioridade inicial: ", heapq_list)
    
    # (linha, coluna)
    # (-1, 0)   diminui uma linha(move para cima) e mantém a coluna
    # (1, 0)    aumenta uma linha(move para baixo) e mantém a coluna
    # (0, -1)   mantém a linha e diminui uma coluna(move para esquerda) 
    # (0, 1)    mantém a linha e aumenta uma coluna(move para direita)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Exploração dos vizinhos
    while heapq_list:
        #retira o nó com menor f(n)  da fila
        cost, node = heapq.heappop(heapq_list)
        print(f"\n\033[1mExplorando o nó \33[41m{node}\33[0m \033[1mcom custo \33[41m{cost}\33[0m")
        print("*************************************")

        #verifica se o nó já foi visitado
        if node in node_visited:
            continue
        else:
            node_visited.add(node)

        #verifica se é o nó objetivo
        if node == goal:
            print("\n\033[42mAchamos o nó objetivo!\33[0m")
            return

        #expanda os nós vizinhos
        for row, col in moves:
            #node[0] = linha, node[1] = coluna
            neighbor = (node[0] + row, node[1] + col)

            #verifica se a linha e coluna estão dentro da matriz
            if not((neighbor[0] > 0 or neighbor[0] < matrix.shape[0]) or (neighbor[1] > 0 or neighbor[1] <= matrix.shape[1])):
                continue
            
            #verifica se é parede
            if matrix[neighbor] == 0:
                continue

            #calcula o g(n) 
            g_node_score = G_path_cost[node] + matrix[neighbor]   

            #verifica se o vizinho já foi visitado e o custo não é melhor, ignore
            if neighbor in node_visited and g_node_score >= G_path_cost.get(neighbor, 0):
                continue
            
            #atualiza o custo do nó e adiciona na fila
            if g_node_score < G_path_cost.get(neighbor, float('inf')):
                G_path_cost[neighbor] = g_node_score
                F_total_cost[neighbor] = g_node_score + manhattan_distance(neighbor, goal)
                heapq.heappush(heapq_list, (F_total_cost[neighbor], neighbor))

                print(f"\33[44mVizinho: {neighbor}\33[0m")
                print(f"g(n) = {G_path_cost[node]} + {matrix[neighbor]} = {g_node_score}")
                print(f"f(n) = {g_node_score} + {manhattan_distance(neighbor, goal)} = {F_total_cost[neighbor]}")
                print("Fila de prioridade: ", heapq_list)
                print("")
   
            time.sleep(10)

image_path = "img/mapa_laboratorio.png"
matrix = proccess_imagem(image_path)

np.savetxt("mapa_laboratorio.txt", matrix, fmt="%d")

A_star_search((6, 40), (41, 40), matrix)