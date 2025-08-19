import argparse
import time
from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from pulp import LpMinimize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum, value
except ImportError as exc:
    raise SystemExit(
        "PuLP is required. Install with: pip install pulp"
    ) from exc


ZC_DEFAULT: complex = 50 + 0j


# -----------------------------
# Attenuation computations
# -----------------------------

def attenuation_final(abcd_matrix: np.ndarray) -> float:
    cosh_gamma_l = abcd_matrix[0, 0]
    attenuation_db = 20 * np.log10(abs(cosh_gamma_l)) - 6.0206
    return float(attenuation_db)


def calculate_attenuation_matrix(gamma: complex, length: float, char_impedance: complex) -> np.ndarray:
    diag = np.cosh(gamma * length)
    base_off_diag = np.sinh(gamma * length)
    return np.array(
        [
            [diag, char_impedance * base_off_diag],
            [base_off_diag / char_impedance, diag],
        ]
    )


def load_inputs(nodes_path: str, edges_path: str, zc_gamma_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.read_excel(nodes_path)
    edges_df = pd.read_excel(edges_path)
    zc_gamma_df = pd.read_csv(zc_gamma_path, delimiter=';')
    return nodes_df, edges_df, zc_gamma_df


def create_graph(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, zc_gamma_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes_df['ID_NODE'])

    zc_gamma_dict: Dict = {
        row['KEY']: (complex(row['rGAMMA'], row['iGAMMA']), complex(row['rZC'], row['iZC']))
        for _, row in zc_gamma_df.iterrows()
    }

    for _, row in edges_df.iterrows():
        from_node = row['ID_FROM_NODE']
        to_node = row['ID_TO_NODE']
        length = float(row['LENGTH'])
        key = row['DIAMETER']

        if key in zc_gamma_dict:
            gamma, zc = zc_gamma_dict[key]
            abcd_matrix = calculate_attenuation_matrix(gamma, length, zc)
            graph.add_edge(from_node, to_node, ABCD=abcd_matrix, gamma=gamma, length=length, char_impedance=zc)

    return graph


def _bfs_and_collect_attenuations(graph: nx.Graph, start_node, index: int, total_nodes: int, verbose: bool) -> List[Tuple[int, int, float]]:
    queue: deque = deque([(start_node, np.eye(2))])
    visited = {start_node}
    attenuations: List[Tuple[int, int, float]] = []

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

    if verbose:
        print(f"Progression: {index}/{total_nodes} nœuds traités.")
    return attenuations


def compute_all_attenuations(graph: nx.Graph, n_jobs: int = -1, verbose: bool = True) -> List[Tuple[int, int, float]]:
    total_nodes = len(graph.nodes())
    results = Parallel(n_jobs=n_jobs)(
        delayed(_bfs_and_collect_attenuations)(graph, node, index, total_nodes, verbose)
        for index, node in enumerate(graph.nodes(), start=1)
    )
    # Flatten
    return [item for sublist in results for item in sublist]


# -----------------------------
# MWVC Solver (PuLP)
# -----------------------------

def compute_node_costs_from_attenuations(graph: nx.Graph, attenuations: Iterable[Tuple[int, int, float]]) -> Dict:
    node_costs: Dict = {node: 0.0 for node in graph.nodes()}
    for start_node, end_node, att_db in attenuations:
        abs_att = abs(att_db)
        node_costs[start_node] += abs_att
        node_costs[end_node] += abs_att
    return node_costs


def normalize_costs(node_costs: Dict) -> Dict:
    if not node_costs:
        return {}
    min_cost = min(node_costs.values())
    max_cost = max(node_costs.values())
    if max_cost == min_cost:
        return {node: 1.0 for node in node_costs}
    return {
        node: 1.0 + 99.0 * (cost - min_cost) / (max_cost - min_cost)
        for node, cost in node_costs.items()
    }


def solve_mwvc(graph: nx.Graph, node_costs: Dict) -> Tuple[Sequence, float]:
    problem = LpProblem("MWVC", LpMinimize)
    x_vars = {node: LpVariable(f"x_{node}", cat='Binary') for node in graph.nodes()}

    problem += lpSum(node_costs[node] * x_vars[node] for node in graph.nodes())

    for u, v in graph.edges():
        problem += x_vars[u] + x_vars[v] >= 1

    problem.solve(PULP_CBC_CMD(msg=False))

    cover = []
    for node in graph.nodes():
        node_val = value(x_vars[node])
        if node_val is None:
            raise RuntimeError(
                f"Le solveur n'a pas trouvé de solution pour le noeud {node}. Vérifiez les coûts et la formulation."
            )
        if node_val > 0.5:
            cover.append(node)

    total_cost = float(sum(node_costs[node] for node in cover))
    return cover, total_cost


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule les atténuations d'un graphe (PLC) puis résout le MWVC. "
            "Entrées: nodes.xlsx, edges.xlsx, zc_gamma_func_cenelec_b.csv"
        )
    )
    parser.add_argument("--nodes", required=True, help="Chemin vers nodes.xlsx")
    parser.add_argument("--edges", required=True, help="Chemin vers edges.xlsx")
    parser.add_argument("--zc-gamma", required=True, help="Chemin vers zc_gamma_func_cenelec_b.csv")
    parser.add_argument("--out-atten", default=None, help="Fichier CSV de sortie pour les atténuations (optionnel)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Nombre de jobs pour le parallélisme (joblib)")
    parser.add_argument("--quiet", action="store_true", help="Réduit la verbosité (pas de progression BFS)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    nodes_df, edges_df, zc_gamma_df = load_inputs(args.nodes, args.edges, args.zc_gamma)

    graph = create_graph(edges_df, nodes_df, zc_gamma_df)

    start_time = time.time()
    attenuations = compute_all_attenuations(graph, n_jobs=args.n_jobs, verbose=not args.quiet)
    elapsed = time.time() - start_time

    print(f"Temps d'exécution total (atténuations): {elapsed:.2f} s")

    if args.out_atten:
        att_df = pd.DataFrame(attenuations, columns=["Start Node", "End Node", "Attenuation (dB)"])
        att_df.to_csv(args.out_atten, index=False)
        print(f"Atténuations écrites dans: {args.out_atten}")

    node_costs_raw = compute_node_costs_from_attenuations(graph, attenuations)
    node_costs = normalize_costs(node_costs_raw)

    cover, total_cost = solve_mwvc(graph, node_costs)

    print(f"Taille de la couverture MWVC : {len(cover)}")
    print(f"Coût total de la couverture : {total_cost:.2f}")


if __name__ == "__main__":
    main() 