import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import time
from itertools import permutations

class Agent:
    def proccess_imagem(self, image_path, regions, group_positions):
        """
            Processa a imagem do mapa para identificar os tipos de terreno e cria uma matriz representando o mapa.

            Args:
                image_path (str): Caminho para a imagem do mapa.
                regions (dict): Dicionário contendo os intervalos HSV e custos associados para cada tipo de terreno.
                group_positions (dict): Dicionário com as posições dos personagens e seus custos.

            Returns:
                numpy.ndarray: Matriz representando o mapa, onde cada célula contém o custo do terreno.
        """
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (42, 42))
        image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        map_matrix = np.zeros((42, 42), dtype=int)

        # processando a imagem para identificar os tipos de terrenos
        for terrain, (lower, upper, cost) in regions.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            # máscara booleana para cada cor
            mask = cv2.inRange(image_hsv, lower, upper)
            # pixels ficam entre 0 e 255, a posição que tem 255, recebe o custo associado
            map_matrix[mask > 0] = cost

        # adicionando o grupo de personagens na matriz
        for name, (position, cost) in group_positions.items():
            map_matrix[position] = cost

        return map_matrix
    
    def manhattan_distance(self, origin, destiny):
        """
            Calcula a distância Manhattan entre dois pontos.

            Args:
                origin (tuple): Coordenadas do ponto de origem (linha, coluna).
                destiny (tuple): Coordenadas do ponto de destino (linha, coluna).

            Returns:
                int: Distância Manhattan entre os dois pontos.

            Observação:
                A distância Manhattan é a soma das diferenças absolutas entre as coordenadas dos pontos.
                Levando em conta que:
                    (x1, y1) e (x2, y2)
                a distância em linha reta entre dois pontos é:    
                |x1 - x2| + |y1 - y2|
        """
        return abs(origin[0] - destiny[0]) + abs(origin[1] - destiny[1])
    
    def print_manhattan_distance_rescue(self, group_positions):
        """
            Imprime a distância Manhattan entre o ponto inicial ('E') e cada personagem a ser resgatado.

            Args:
                group_positions (dict): Dicionário com as posições dos personagens.
        """
        start = (6, 40)

        for name, (position, _) in group_positions.items():
            if name == "E":
                continue

            distance = self.manhattan_distance(start, position)
            print(f"Distância Manhattan de E até a {name}: {distance}")

    def node_path(self, path_to_goal, node):
        """
            Reconstrói o caminho do nó inicial até o nó objetivo.

            Args:
                path_to_goal (dict): Dicionário que mapeia cada nó para o nó anterior no caminho.
                node (tuple): Nó objetivo.

            Returns:
                list: Lista de nós representando o caminho do nó inicial até o nó objetivo.

            Observação:
                O caminho é reconstruído a partir do nó objetivo, retrocedendo até o nó inicial.
        """
        path = [node] 

        while node in path_to_goal:
            node = path_to_goal[node] #nó anteior
            path.append(node) #adiciona o nó anterior
        return path[::-1]
    
    def pairs_path_cost(self, matrix, positions):
        """
            Calcula o custo do menor caminho entre todos os pares de nós.

            Args:
                matrix (numpy.ndarray): Matriz representando o mapa.
                positions (dict): Dicionário com as posições dos nós.

            Returns:
                dict: Dicionário com os custos entre os pares de nós.
        """
        costs = {}
        for name, (pos, _) in positions.items():
            for name2, (pos2, _) in positions.items():
                if name == name2:
                    continue
                cost, path = self.A_star_search(pos, pos2, matrix)
                costs[(name, name2)] = cost
                #print(f"Custo de {name} até {name2}: {cost}")
        return costs
    
    def find_best_path(self, matrix, positions):
        """
            Encontra o melhor caminho que passa por todos os personagens e retorna o caminho completo.

            Args:
                matrix (numpy.ndarray): Matriz representando o mapa.
                positions (dict): Dicionário com as posições dos personagens.

            Returns:
                list: Lista de coordenadas representando o caminho completo.

            Observação:
                utiliza-se permutações para encontrar o melhor caminho que passa por todos os personagens.
        """
        nodes_to_explore = {}
        start = 'E'
        end = 'X'
        current_cost = 0
        best_path = None
        min_cost = float('inf')

        pairs_cost = self.pairs_path_cost(matrix, positions)

        for name, (pos, _) in positions.items():
            if name == 'E' or name == 'X':
                continue
            nodes_to_explore[name] = pos

        # permutations() -> retorna todos os possíveis pares de elementos
        for perm in permutations(nodes_to_explore.keys()):
            current_path = [start] + list(perm) + [end]
            #print(current_path)
            #print(pairs_cost)
            for character in range(len(current_path) -1):
                # acumulando o custo de cada nó do caminho
                current_cost += pairs_cost[(current_path[character], current_path[character+1])]
            
            if current_cost < min_cost:
                min_cost = current_cost
                best_path = current_path

        full_path = []
        for i in range(len(best_path) - 1):
            start_node = positions[best_path[i]][0]
            end_node = positions[best_path[i + 1]][0]
            _, path = self.A_star_search(start_node, end_node, matrix)
            if full_path:
                full_path.extend(path[1:])  # evita duplicar o último nó do subcaminho
            else:
                full_path.extend(path)

        print(f"\n\33[92mMelhor caminho: {best_path}\33[0m")
        print(f"\33[92mCusto do melhor caminho: {min_cost}\33[0m")
        print(f"Caminho completo: \n{full_path}")

        return full_path
    
    def A_star_search(self, start, goal, matrix):
        """
            Implementa o algoritmo A* para encontrar o menor caminho entre dois nós.

            Args:
                start (tuple): Coordenadas do nó inicial (linha, coluna).
                goal (tuple): Coordenadas do nó objetivo (linha, coluna).
                matrix (numpy.ndarray): Matriz representando o mapa.

            Returns:
                tuple: Custo total do caminho e o caminho como uma lista de coordenadas.
        """
        # Inicialização 
        heapq_list = []
        heapq.heappush(heapq_list, (0, start))
        G_path_cost = {start: 0}
        # a função de avaliação inicialmente é definida como a distância de manhattan do nó inicial
        F_total_cost = {start: self.manhattan_distance(start, goal)}
        node_visited = set() 
        path_to_goal = {}

        print("\033[1mg(n) inicial: ", G_path_cost)
        print("\033[1mf(n) inicial: ", F_total_cost)
        print("\033[1mFila de prioridade inicial: ", heapq_list)
        print()
        '''
            (linha, coluna)
            (-1, 0)   diminui uma linha(move para cima) e mantém a coluna
            (1, 0)    aumenta uma linha(move para baixo) e mantém a coluna
            (0, -1)   mantém a linha e diminui uma coluna(move para esquerda) 
            (0, 1)    mantém a linha e aumenta uma coluna(move para direita)'
        '''
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Exploração dos vizinhos
        while heapq_list:
            # retira o nó com menor f(n)  da fila  
            cost, node = heapq.heappop(heapq_list)
            print (
                f"\n\33[40m{'*' * 37}\033[0m\n"
                f"\33[40m\033[1mExplorando o nó \033[96m{node}\033[0m\33[40m\033[1m com custo \033[96m{cost}\033[0m\n"
                f"\33[40m{'*' * 37}\033[0m"
            )

            # verifica se o nó já foi visitado
            if node in node_visited:
                continue
            else:
                node_visited.add(node)

            # verifica se é o nó objetivo
            if node == goal:
                path_cost = 0
                print(f"\n\033[42mAchamos o nó objetivo!\33[0m")
                print(f"\nQuantidade de nós percorridos: {len(node_visited)}")
                path = self.node_path(path_to_goal, node)
                #print(f"\nCaminho de {start} até {node}:\n{path}")

                for node in path:
                    row = node[0]
                    col = node[1]
                    path_cost += matrix[row][col]

                return path_cost, path

            # Expande os nós vizinhos
            for row, col in moves:
                # node[0] = linha, node[1] = coluna
                neighbor = (node[0] + row, node[1] + col)

                # verifica se a linha e coluna estão dentro da matriz
                if not((0 <= neighbor[0] <= matrix.shape[0] - 1) and (0 <= neighbor[1] <= matrix.shape[1] - 1)):
                    print(f"\n\33[31mVizinho {neighbor} fora dos limites\33[0m")
                    continue
                
                # verifica se é parede
                if matrix[neighbor] == 0:
                    print(f"\n\33[31mVizinho {neighbor} é parede\33[0m")
                    continue

                # calcula o g(n) 
                g_node_score = G_path_cost[node] + matrix[neighbor]   

                # verifica se o vizinho já foi visitado e o custo não é melhor, ignore
                if neighbor in node_visited and g_node_score >= G_path_cost.get(neighbor, 0):
                    print(f"\n\33[31mVizinho {neighbor} já visitado ou custo não é melhor\33[0m")
                    continue
                
                # Atualiza o custo do nó e adiciona na fila
                if g_node_score < G_path_cost.get(neighbor, float('inf')):
                    G_path_cost[neighbor] = g_node_score
                    F_total_cost[neighbor] = g_node_score + self.manhattan_distance(neighbor, goal)
                    heapq.heappush(heapq_list, (F_total_cost[neighbor], neighbor))
                    # o nó atual (node) é registrado como o nó anterior do vizinho (neighbor).
                    path_to_goal[neighbor] = node

                    print(f"\n\33[34mVizinho: {neighbor}\33[0m")
                    print(f"g(n) = {G_path_cost[node]} + {matrix[neighbor]} = {g_node_score}")
                    print(f"f(n) = {g_node_score} + {self.manhattan_distance(neighbor, goal)} = {F_total_cost[neighbor]}")
                    print("Fila de prioridade: ", heapq_list)
            #time.sleep(2)

    def visualize_agent_moves(self, matrix, path, group_positions):
        """
            Visualiza o movimento do agente no mapa com uma animação.

            Args:
                matrix (numpy.ndarray): Matriz representando o mapa.
                path (list): Lista de coordenadas representando o caminho do agente.
                group_positions (dict): Dicionário com as posições dos personagens.
        """
        matrix_copy = matrix.copy()

        fig, ax = plt.subplots(figsize=(8, 8))
        img = ax.imshow(matrix_copy, cmap="gist_gray", interpolation="nearest")
        ax.axis("off")

        # adiciona os personagens no mapa com marcadores
        for name, (position, _) in group_positions.items():
            if name not in ["E", "X"]: 
                row, col = position
                ax.scatter(col, row, color="green", label=name, marker="o", s=100) 

        # atualiza a posição do agente
        def update(frame):
            if frame > 0:
                # restaura o valor original da célula anterior
                prev_row, prev_col = path[frame - 1]
                matrix_copy[prev_row, prev_col] = matrix[prev_row, prev_col]

            # atualiza a posição atual do agente
            row, col = path[frame]
            matrix_copy[row, col] = -1  
            ax.scatter(col, row, color="red", label=name, marker="s", s=100)

            # atualiza a imagem exibida
            img.set_data(matrix_copy)
            return [img]

        ani = animation.FuncAnimation(
            fig, update, frames=len(path), interval=100, blit=False
        )

        plt.title("Caminho do agente")
        plt.show()