# -*- coding: utf-8 -*-
"""
MWVC via QAOA (Qiskit 2.x) on a 10-node NetworkX graph.
Formulation QUBO/Ising basée sur Glover–Kochenberger–Du (2022).

Prérequis:
    pip install networkx qiskit qiskit-aer

Remarque:
- On utilise un simulateur (Estimator/Sampler) pour QAOA. Le code
  reste identique si vous exécutez sur du matériel quantique.
"""

import math
import random
from itertools import product

import networkx as nx
import numpy as np

# Qiskit 2.x
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit.circuit import Parameter
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
from qiskit_aer import Aer

# -------------------------
# 1) Graphe & poids (10 nœuds)
# -------------------------
def build_graph_10():
    G = nx.Graph()
    G.add_nodes_from(range(10))
    # Graphe connecté, pas trop dense
    edges = [
        (0, 1), (0, 3),
        (1, 2), (1, 4),
        (2, 5),
        (3, 4), (3, 6),
        (4, 5), (4, 7),
        (5, 8),
        (6, 7),
        (7, 8), (7, 9),
        (8, 9),
    ]
    G.add_edges_from(edges)
    # Poids positifs (MWVC)
    rng = random.Random(42)
    w = {i: rng.randint(1, 5) for i in G.nodes()}
    return G, w


# -------------------------
# 2) QUBO -> Ising (pour QAOA)
#    Pénalités: P*(1 - x_i - x_j + x_i x_j)
#    Ising: x_i = (1 - s_i)/2  avec s_i in {-1, 1}
#    => H = sum_{(i,j)} (P/4) Z_i Z_j + sum_i ((P/4)deg(i) - w_i/2) Z_i + const
# -------------------------
def ising_params_mwvc(G, w, P):
    deg = dict(G.degree())
    J = {}  # couplages ZZ
    h = {i: (P / 4.0) * deg[i] - w[i] / 2.0 for i in G.nodes()}
    for i, j in G.edges():
        a, b = sorted((i, j))
        J[(a, b)] = J.get((a, b), 0.0) + P / 4.0
    return J, h


def cost_operator_from_Jh(num_qubits, J, h):
    """
    Construit l'opérateur de coût SparsePauliOp à partir de:
      H = sum_{i<j} J_ij Z_i Z_j + sum_i h_i Z_i  (+ const)
    en utilisant from_sparse_list (plus sûr que des chaînes 'Z...Z').
    """
    terms = []
    for (i, j), coeff in J.items():
        terms.append(("ZZ", [i, j], float(coeff)))
    for i, coeff in h.items():
        terms.append(("Z", [i], float(coeff)))
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=num_qubits)
    return op


# -------------------------
# 3) Outils MWVC: faisabilité, coût, réparation
# -------------------------
def is_feasible_mwvc(G, x_vec):
    """Vérifie x_i + x_j >= 1 pour toutes les arêtes."""
    for i, j in G.edges():
        if x_vec[i] + x_vec[j] < 1:
            return False
    return True


def weight_cost(w, x_vec):
    return sum(w[i] * x_vec[i] for i in x_vec)


def greedy_repair(G, w, x_vec):
    """
    Répare un bitstring en ajoutant le sommet le moins cher
    pour chaque arête non couverte (opération locale).
    """
    x = x_vec.copy()
    for i, j in G.edges():
        if x[i] + x[j] == 0:
            # ajoute le moins cher
            if w[i] <= w[j]:
                x[i] = 1
            else:
                x[j] = 1
    return x


# -------------------------
# 4) QAOA: optimisation + échantillonnage
# -------------------------
def qaoa_mwvc(G, w, P=0.0, reps=2, shots=4096, maxiter=200, seed=1234):
    n = G.number_of_nodes()
    if P <= 0:
        # Choix de P (règle simple, robuste en pratique)
        P = 2.0 * max(w.values())

    # Paramètres Ising et opérateur de coût
    J, h = ising_params_mwvc(G, w, P)
    Hc = cost_operator_from_Jh(n, J, h)

    # Construction de l'ansatz QAOA
    ansatz = QAOAAnsatz(cost_operator=Hc, reps=reps, flatten=True)
    
    # Paramètres initiaux aléatoires
    initial_point = np.random.random(2 * reps)
    
    # Backend et primitives
    backend = Aer.get_backend('qasm_simulator')
    estimator = BackendEstimatorV2(backend=backend)
    sampler = BackendSamplerV2(backend=backend)
    
    # Fonction objectif pour l'optimisation
    def objective(params):
        circuit = ansatz.assign_parameters(params)
        job = estimator.run([(circuit, Hc)])
        result = job.result()
        return result[0].real
    
    # Optimisation
    result = minimize(objective, initial_point, method='COBYLA', 
                     options={'maxiter': maxiter, 'rhobeg': 0.5, 'tol': 1e-3})
    opt_point = result.x
    
    # Échantillonnage de l'ansatz à paramètres optimaux
    assigned = ansatz.assign_parameters(opt_point, inplace=False)
    job = sampler.run(assigned, shots=shots)
    quasi = job.result().quasi_dists[0]

    # Trie des bitstrings par probabilité décroissante
    items = sorted(quasi.items(), key=lambda kv: kv[1], reverse=True)

    # Sélection: meilleur bitstring faisable (avec ou sans réparation)
    best = None
    best_cost = math.inf
    best_raw = None
    nodes = list(range(n))

    def bitstring_to_x(bstr):
        # Qiskit renvoie généralement des bitstrings little-endian (qubit 0 = bit de droite).
        # On mappe qubit i -> noeud i via bit[n-1-i].
        x = {}
        for i in nodes:
            bit = int(bstr[-1 - i])
            x[i] = bit
        return x

    # Parcours des échantillons
    for bstr, prob in items:
        # bstr peut être une int->proba si quasi-dist compacte; normalisons en str binaire longueur n
        if isinstance(bstr, int):
            bstr = format(bstr, f"0{n}b")
        x = bitstring_to_x(bstr)
        if is_feasible_mwvc(G, x):
            c = weight_cost(w, x)
            if c < best_cost:
                best = x
                best_cost = c
                best_raw = bstr
                break

    # Si rien de faisable directement, essaye la réparation gloutonne
    if best is None and items:
        bstr0, _ = items[0]
        if isinstance(bstr0, int):
            bstr0 = format(bstr0, f"0{n}b")
        x0 = bitstring_to_x(bstr0)
        xr = greedy_repair(G, w, x0)
        if is_feasible_mwvc(G, xr):
            best = xr
            best_cost = weight_cost(w, xr)
            best_raw = bstr0

    return {
        "best_x": best,
        "best_cost": best_cost,
        "best_raw_bitstring": best_raw,
        "quasi_dist": items[:20],  # top-20 pour inspection
        "P": P,
        "reps": reps,
        "opt_energy": float(result.fun),
        "opt_point": np.array(opt_point).tolist(),
    }


# -------------------------
# 5) Vérification exacte (brute force 2^n) pour n=10
# -------------------------
def brute_force_mwvc(G, w):
    n = G.number_of_nodes()
    best = None
    best_cost = math.inf
    for bits in product([0, 1], repeat=n):
        x = {i: bits[i] for i in range(n)}
        if is_feasible_mwvc(G, x):
            c = weight_cost(w, x)
            if c < best_cost:
                best_cost = c
                best = x
    return best, best_cost


# -------------------------
# 6) Exécution
# -------------------------
if __name__ == "__main__":
    G, w = build_graph_10()
    print("Edges:", sorted(G.edges()))
    print("Weights:", w)

    res = qaoa_mwvc(G, w, P=0.0, reps=2, shots=4096, maxiter=200, seed=7)
    print("\n=== QAOA result ===")
    print("Penalty P:", res["P"])
    print("QAOA layers (reps):", res["reps"])
    print("Optimal energy (Ising):", res["opt_energy"])
    print("Best raw bitstring (little-endian):", res["best_raw_bitstring"])
    print("QAOA best feasible cover:", res["best_x"])
    print("QAOA best feasible cost (sum w_i x_i):", res["best_cost"])

    # Validation (exacte)
    x_star, c_star = brute_force_mwvc(G, w)
    print("\n=== Exact (brute force) ===")
    print("Optimal cover:", x_star)
    print("Optimal cost:", c_star)

    # Écart éventuel
    if res["best_x"] is not None:
        gap = res["best_cost"] - c_star
        print("\nGap (QAOA - exact):", gap)
    else:
        print("\nQAOA n'a pas produit de solution faisable directement.")