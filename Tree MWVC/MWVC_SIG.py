import numpy as np
import networkx as nx
import pandas as pd
from pulp import *
import matplotlib.pyplot as plt

# 1. Charger les données du graphe
nodes_df = pd.read_excel('Experiment/nodes.xlsx')
edges_df = pd.read_excel('Experiment/edges.xlsx')

# 2. Construire le graphe
G = nx.Graph()
G.add_nodes_from(nodes_df['ID_NODE'])
for _, row in edges_df.iterrows():
    G.add_edge(row['ID_FROM_NODE'], row['ID_TO_NODE'])

# 3. Charger les atténuations
att_df = pd.read_csv('Experiment/attenuations_bfs.csv')

# 4. Calculer les coûts des nœuds à partir des atténuations 
node_costs = {node: 0 for node in G.nodes()}
for _, row in att_df.iterrows():
    u, v, att = row['Start Node'], row['End Node'], row['Attenuation (dB)']
    node_costs[u] += abs(att)
    node_costs[v] += abs(att)  # On prend la valeur absolue pour éviter les coûts négatifs

# Normalisation des coûts (entre 1 et 100)
min_cost = min(node_costs.values())
max_cost = max(node_costs.values())
if max_cost == min_cost:
    for node in node_costs:
        node_costs[node] = 1
else:
    for node in node_costs:
        node_costs[node] = 1 + 99 * (node_costs[node] - min_cost) / (max_cost - min_cost)

# Résolution MWVC
prob = LpProblem("MWVC", LpMinimize)
x = {node: LpVariable(f'x_{node}', cat='Binary') for node in G.nodes()}
prob += lpSum(node_costs[node] * x[node] for node in G.nodes())
for u, v in G.edges():
    prob += x[u] + x[v] >= 1
prob.solve(PULP_CBC_CMD(msg=False))

# Vérification de la solution
cover = set()
for node in G.nodes():
    val = value(x[node])
    if val is None:
        raise RuntimeError(f"Le solveur n'a pas trouvé de solution pour le noeud {node}. Vérifiez les coûts et la formulation.")
    if val > 0.5:
        cover.add(node)

# 6. Affichage des résultats
print(f"Taille de la couverture MWVC : {len(cover)}")
print(f"Coût total de la couverture : {sum(node_costs[node] for node in cover):.2f}")

# 7. Visualisation (optionnelle)
def visualize_graph(graph, node_costs, cover=None, title="MWVC Solution"):
    pos = nx.spring_layout(graph, k=2, iterations=50)
    plt.figure(figsize=(15, 12))
    node_colors = ['red' if cover and node in cover else 'skyblue' for node in graph.nodes()]
    node_sizes = [100 + 20 * node_costs[node] for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5)
    labels = {node: f"{node}\n({node_costs[node]:.1f})" for node in graph.nodes() if cover and node in cover}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    plt.title(title, fontsize=14, pad=20)
    plt.axis('off')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Nœud dans la couverture', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Nœud hors couverture', markerfacecolor='skyblue', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# Décommentez pour visualiser
visualize_graph(G, node_costs, cover, title="Solution MWVC réelle (SIG)") 