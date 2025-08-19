import numpy as np
import networkx as nx
from collections import deque
import random
import matplotlib.pyplot as plt
from pulp import *
import time

def get_random_attenuation_matrix(gamma, length, char_impedance):
    """Génère une matrice d'atténuation aléatoire ABCD."""
    diag = np.cosh(gamma * length)
    base_off_diag = np.sinh(gamma * length)
    return np.array([[diag, char_impedance * base_off_diag], 
                     [base_off_diag / char_impedance, diag]])

def generate_plc_network(num_nodes, edge_prob=0.3):
    """Génère un graphe général (non arbre) avec des matrices d'atténuation aléatoires."""
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    
    # Assure la connectivité
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        while len(components) > 1:
            u = random.choice(list(components[0]))
            v = random.choice(list(components[1]))
            G.add_edge(u, v)
            components = list(nx.connected_components(G))
    
    for u, v in G.edges():
        length = np.random.uniform(0.1, 1.0)
        gamma = np.random.uniform(0.01, 0.1) + 1j * np.random.uniform(0.01, 0.1)
        char_impedance = 50 * 1j
        G[u][v]["ABCD"] = get_random_attenuation_matrix(gamma, length, char_impedance)
    return G

def attenuation_final(matrix):
    """Calcule l'atténuation finale à partir de la matrice ABCD."""
    cosh_gamma_l = matrix[0, 0]
    return 20 * np.log10(abs(cosh_gamma_l)) - 6.0206

def calculate_all_attenuations(graph):
    """Calcule toutes les atténuations entre toutes les paires de nœuds."""
    attenuations = []
    for start_node in graph.nodes():
        queue = deque([(start_node, np.eye(2))])
        visited = {start_node}
        while queue:
            current_node, current_matrix = queue.popleft()
            if current_node != start_node:
                final_att = attenuation_final(current_matrix)
                attenuations.append((start_node, current_node, final_att))
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    new_matrix = np.dot(current_matrix, graph[current_node][neighbor]['ABCD'])
                    queue.append((neighbor, new_matrix))
                    visited.add(neighbor)
    return attenuations

def assign_node_costs(graph, attenuations, threshold=30):
    """Assigne des coûts aux nœuds basés sur les atténuations."""
    node_costs = {node: 0 for node in graph.nodes()}
    
    # Calcul des coûts de base basés sur les atténuations
    for u, v, att in attenuations:
        if att > threshold:  # Si l'atténuation dépasse le seuil
            node_costs[u] += att - threshold
            node_costs[v] += att - threshold
    
    # Ajout d'une variation aléatoire pour chaque nœud
    for node in node_costs:
        # Ajoute une variation aléatoire entre 0.5 et 1.5
        random_factor = np.random.uniform(0.5, 1.5)
        node_costs[node] *= random_factor
    
    # Normalisation des coûts
    max_cost = max(node_costs.values()) if node_costs.values() else 1
    if max_cost == 0:  # Si tous les coûts sont à 0
        for node in node_costs:
            # Attribue un coût aléatoire entre 1 et 10
            node_costs[node] = np.random.uniform(1, 10)
    else:
        for node in node_costs:
            # Normalise entre 1 et 10 avec une variation aléatoire
            base_cost = 1 + (node_costs[node] / max_cost) * 9
            variation = np.random.uniform(-0.5, 0.5)  # Variation de ±0.5
            node_costs[node] = max(1, min(10, base_cost + variation))  # Garde entre 1 et 10
    
    return node_costs

def refined_greedy_mwvc(graph, node_costs):
    """Algorithme Refined Greedy pour MWVC."""
    cover = set()
    uncovered_edges = set(graph.edges())
    
    while uncovered_edges:
        # Calcul du ratio coût/degré pour chaque nœud non couvert
        ratios = {}
        for node in graph.nodes():
            if node not in cover:
                degree = sum(1 for edge in uncovered_edges if node in edge)
                if degree > 0:
                    ratios[node] = node_costs[node] / degree
        
        if not ratios:
            break
            
        # Sélection du nœud avec le meilleur ratio
        selected_node = min(ratios.keys(), key=lambda x: ratios[x])
        cover.add(selected_node)
        
        # Retirer les edges couverts
        uncovered_edges = {edge for edge in uncovered_edges if selected_node not in edge}
    
    return cover

def local_2_approx_mwvc(graph, node_costs):
    """Algorithme Local 2-Approximation pour MWVC."""
    cover = set()
    uncovered_edges = set(graph.edges())
    
    while uncovered_edges:
        # Prendre un edge non couvert arbitraire
        u, v = next(iter(uncovered_edges))
        
        # Calculer le ratio coût/degré pour les deux nœuds
        degree_u = sum(1 for edge in uncovered_edges if u in edge)
        degree_v = sum(1 for edge in uncovered_edges if v in edge)
        
        ratio_u = node_costs[u] / degree_u if degree_u > 0 else float('inf')
        ratio_v = node_costs[v] / degree_v if degree_v > 0 else float('inf')
        
        # Ajouter le nœud avec le meilleur ratio
        if ratio_u <= ratio_v:
            cover.add(u)
            # Retirer tous les edges couverts par u
            uncovered_edges = {edge for edge in uncovered_edges if u not in edge}
        else:
            cover.add(v)
            # Retirer tous les edges couverts par v
            uncovered_edges = {edge for edge in uncovered_edges if v not in edge}
    
    return cover

def exact_mwvc(graph, node_costs, time_limit=5):
    """Résolution exacte du MWVC en utilisant la programmation linéaire."""
    # Créer le problème
    prob = LpProblem("MWVC", LpMinimize)
    
    # Variables de décision (1 si le nœud est dans la couverture, 0 sinon)
    x = {node: LpVariable(f'x_{node}', cat='Binary') for node in graph.nodes()}
    
    # Fonction objectif
    prob += lpSum(node_costs[node] * x[node] for node in graph.nodes())
    
    # Contraintes
    for u, v in graph.edges():
        prob += x[u] + x[v] >= 1  # Au moins un des deux nœuds doit être dans la couverture
    
    # Résoudre le problème avec limite de temps
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    
    # Récupérer la solution
    cover = {node for node in graph.nodes() if value(x[node]) > 0.5}
    return cover

def evaluate_cover(cover, node_costs):
    """Évalue le coût d'une couverture."""
    return sum(node_costs[node] for node in cover)

def visualize_graph(graph, node_costs, cover=None, title="Original Graph with Costs"):
    """Visualise le graphe avec les coûts et la couverture."""
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 8))
    
    # Dessiner les nœuds
    node_colors = ['red' if cover and node in cover else 'skyblue' for node in graph.nodes()]
    node_sizes = [300 + 50 * node_costs[node] for node in graph.nodes()]
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes)
    
    # Dessiner les edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    
    # Ajouter les labels avec les coûts
    labels = {node: f"{node}\n({node_costs[node]:.1f})" for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=10)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

def run_experiment(num_nodes, edge_prob=0.3):
    """Exécute une expérience pour une taille de graphe donnée."""
    # Génération du graphe
    G = generate_plc_network(num_nodes, edge_prob)
    attenuations = calculate_all_attenuations(G)
    node_costs = assign_node_costs(G, attenuations)
    
    results = {}
    
    # Mesure du temps et des performances pour chaque algorithme
    algorithms = {
        'Exact': lambda g, c: exact_mwvc(g, c, time_limit=5 if num_nodes >= 100 else None),
        'Refined Greedy': refined_greedy_mwvc,
        'Local 2-Approx': local_2_approx_mwvc
    }
    
    for name, algo in algorithms.items():
        start_time = time.time()
        cover = algo(G, node_costs)
        execution_time = time.time() - start_time
        
        results[name] = {
            'time': execution_time,
            'cover_size': len(cover),
            'cover_weight': evaluate_cover(cover, node_costs)
        }
    
    return results

def plot_results(all_results, sizes):
    """Crée des graphiques comparatifs des résultats."""
    algorithms = list(all_results[sizes[0]].keys())
    
    # Préparation des données
    times = {algo: [all_results[size][algo]['time'] for size in sizes] for algo in algorithms}
    cover_sizes = {algo: [all_results[size][algo]['cover_size'] for size in sizes] for algo in algorithms}
    cover_weights = {algo: [all_results[size][algo]['cover_weight'] for size in sizes] for algo in algorithms}
    
    # Création des graphiques
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Temps d'exécution
    for algo in algorithms:
        ax1.plot(sizes, times[algo], marker='o', label=algo)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time')
    ax1.legend()
    ax1.grid(True)
    
    # Taille de la couverture
    for algo in algorithms:
        ax2.plot(sizes, cover_sizes[algo], marker='o', label=algo)
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Cover Size')
    ax2.set_title('Cover Size')
    ax2.legend()
    ax2.grid(True)
    
    # Poids de la couverture
    for algo in algorithms:
        ax3.plot(sizes, cover_weights[algo], marker='o', label=algo)
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Cover Weight')
    ax3.set_title('Cover Weight')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Tailles de graphes à tester
    sizes = [10, 50, 100, 1000]  # Ajout de 1000 nœuds
    all_results = {}
    
    # Exécution des expériences
    for size in sizes:
        print(f"\nTesting with {size} nodes...")
        results = run_experiment(size)
        all_results[size] = results
        
        # Affichage des résultats pour cette taille
        print(f"\nResults for {size} nodes:")
        for algo, metrics in results.items():
            print(f"{algo}:")
            print(f"  Time: {metrics['time']:.3f}s")
            print(f"  Cover Size: {metrics['cover_size']}")
            print(f"  Cover Weight: {metrics['cover_weight']:.2f}")
    
    # Création des graphiques comparatifs
    plot_results(all_results, sizes)

if __name__ == "__main__":
    main()