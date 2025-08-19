import io
import os
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# I/O helpers
# -----------------------------

def read_table_auto(source) -> pd.DataFrame:
    try:
        # If it's an uploaded file-like object, pandas can read directly
        name = getattr(source, "name", None)
        if isinstance(source, (str, os.PathLike)):
            lower = str(source).lower()
        else:
            lower = str(name).lower() if name else ""

        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            return pd.read_excel(source)
        # Try auto-detect CSV separator
        try:
            return pd.read_csv(source, sep=None, engine="python")
        except Exception:
            try:
                return pd.read_csv(source, delimiter=";")
            except Exception:
                return pd.read_csv(source)
    except Exception as exc:
        raise RuntimeError(f"Error reading file '{source}': {exc}")


def load_inputs(nodes_src, edges_src, zc_gamma_src) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes_df = read_table_auto(nodes_src)
    edges_df = read_table_auto(edges_src)
    zc_gamma_df = read_table_auto(zc_gamma_src)
    return nodes_df, edges_df, zc_gamma_df


# -----------------------------
# Attenuations (same core logic as script)
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


def pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    available = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in available:
            return available[cand.lower()]
    return None


def create_graph(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, zc_gamma_df: pd.DataFrame, logs: List[str]) -> nx.Graph:
    graph = nx.Graph()

    node_id_col = pick_col(nodes_df, ["ID_NODE", "NODE_ID", "ID", "NODE"])
    if node_id_col is None:
        raise ValueError("Could not find column ID_NODE in nodes.")
    graph.add_nodes_from(nodes_df[node_id_col])

    key_col = pick_col(zc_gamma_df, ["KEY", "DIAMETER", "K"])  # key used to map line parameters
    r_gamma_col = pick_col(zc_gamma_df, ["rGAMMA", "RGAMMA", "r_gamma", "RG"]) 
    i_gamma_col = pick_col(zc_gamma_df, ["iGAMMA", "IGAMMA", "i_gamma", "IG"]) 
    r_zc_col = pick_col(zc_gamma_df, ["rZC", "RZC", "r_zc", "RZ"]) 
    i_zc_col = pick_col(zc_gamma_df, ["iZC", "IZC", "i_zc", "IZ"]) 

    zc_gamma_dict: Dict = {}
    if key_col and r_gamma_col and i_gamma_col and r_zc_col and i_zc_col:
        for _, row in zc_gamma_df.iterrows():
            try:
                gamma = complex(float(row[r_gamma_col]), float(row[i_gamma_col]))
                zc = complex(float(row[r_zc_col]), float(row[i_zc_col]))
                zc_gamma_dict[row[key_col]] = (gamma, zc)
            except Exception:
                continue
    else:
        logs.append("Warning: missing KEY/rGAMMA/iGAMMA/rZC/iZC columns in ZC/GAMMA file. Edges without parameters will be treated as no attenuation (identity matrix).")

    from_col = pick_col(edges_df, ["ID_FROM_NODE", "FROM", "SRC", "SOURCE"])
    to_col = pick_col(edges_df, ["ID_TO_NODE", "TO", "DST", "TARGET"])
    length_col = pick_col(edges_df, ["LENGTH", "LEN", "L"])
    edge_key_col = pick_col(edges_df, ["DIAMETER", "KEY", "K"])

    if from_col is None or to_col is None:
        raise ValueError("Could not find edge columns (ID_FROM_NODE / ID_TO_NODE).")

    for _, row in edges_df.iterrows():
        from_node = row[from_col]
        to_node = row[to_col]
        length = float(row[length_col]) if length_col and not pd.isna(row[length_col]) else 0.0
        key = row[edge_key_col] if edge_key_col and edge_key_col in row else None

        abcd_matrix = None
        if key is not None and key in zc_gamma_dict:
            gamma, zc = zc_gamma_dict[key]
            try:
                abcd_matrix = calculate_attenuation_matrix(gamma, length, zc)
            except Exception:
                abcd_matrix = None

        if abcd_matrix is None:
            abcd_matrix = np.eye(2)
        graph.add_edge(from_node, to_node, ABCD=abcd_matrix, length=length, key=key)

    return graph


def bfs_collect(graph: nx.Graph, start_node) -> List[Tuple[int, int, float]]:
    queue_nodes: deque = deque([(start_node, np.eye(2))])
    visited = {start_node}
    pairs: List[Tuple[int, int, float]] = []
    while queue_nodes:
        current, current_matrix = queue_nodes.popleft()
        if current != start_node:
            att = attenuation_final(current_matrix)
            pairs.append((start_node, current, att))
        for neigh in graph[current]:
            if neigh not in visited:
                new_matrix = np.dot(current_matrix, graph[current][neigh]["ABCD"])
                queue_nodes.append((neigh, new_matrix))
                visited.add(neigh)
    return pairs


def compute_all_attenuations(graph: nx.Graph) -> List[Tuple[int, int, float]]:
    all_pairs: List[Tuple[int, int, float]] = []
    nodes_list = list(graph.nodes())
    prog = st.progress(0, text="Computing attenuations…")
    total = len(nodes_list)
    for idx, node in enumerate(nodes_list, start=1):
        pairs = bfs_collect(graph, node)
        all_pairs.extend(pairs)
        prog.progress(int(100 * idx / max(1, total)), text=f"Computing attenuations… {idx}/{total}")
    prog.empty()
    return all_pairs


# -----------------------------
# MWVC with PuLP
# -----------------------------

def compute_node_costs_from_attenuations(graph: nx.Graph, attenuations: Iterable[Tuple[int, int, float]]) -> Dict:
    node_costs: Dict = {node: 0.0 for node in graph.nodes()}
    for u, v, att_db in attenuations:
        val = abs(att_db)
        node_costs[u] += val
        node_costs[v] += val
    return node_costs


def normalize_costs(node_costs: Dict) -> Dict:
    if not node_costs:
        return {}
    min_cost = min(node_costs.values())
    max_cost = max(node_costs.values())
    if max_cost == min_cost:
        return {n: 1.0 for n in node_costs}
    return {n: 1.0 + 99.0 * (c - min_cost) / (max_cost - min_cost) for n, c in node_costs.items()}


def solve_mwvc(graph: nx.Graph, node_costs: Dict) -> Tuple[List, float]:
    try:
        from pulp import LpMinimize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum, value
    except Exception as exc:
        raise RuntimeError("PuLP is not installed. Install with: pip install pulp") from exc

    problem = LpProblem("MWVC", LpMinimize)
    x = {node: LpVariable(f"x_{node}", cat="Binary") for node in graph.nodes()}

    problem += lpSum(node_costs[node] * x[node] for node in graph.nodes())
    for u, v in graph.edges():
        problem += x[u] + x[v] >= 1

    problem.solve(PULP_CBC_CMD(msg=False))

    cover: List = []
    for node in graph.nodes():
        val = value(x[node])
        if val is None:
            raise RuntimeError(
                f"Solver did not return a value for node {node}. Check costs and formulation."
            )
        if val > 0.5:
            cover.append(node)
    total_cost = float(sum(node_costs[n] for n in cover))
    return cover, total_cost


def local_2_approx_mwvc(graph: nx.Graph, node_costs: Dict) -> List:
    """Local 2-Approximation heuristic for MWVC.

    Iteratively pick an uncovered edge (u, v), compute cost/degree ratio for u and v
    w.r.t the set of still-uncovered edges, and add the better node to the cover.
    """
    cover: set = set()
    uncovered_edges: set = set(graph.edges())

    while uncovered_edges:
        u, v = next(iter(uncovered_edges))

        degree_u = sum(1 for e in uncovered_edges if u in e)
        degree_v = sum(1 for e in uncovered_edges if v in e)

        ratio_u = (node_costs.get(u, 0.0) / degree_u) if degree_u > 0 else float("inf")
        ratio_v = (node_costs.get(v, 0.0) / degree_v) if degree_v > 0 else float("inf")

        if ratio_u <= ratio_v:
            cover.add(u)
            uncovered_edges = {e for e in uncovered_edges if u not in e}
        else:
            cover.add(v)
            uncovered_edges = {e for e in uncovered_edges if v not in e}

    return list(cover)


# -----------------------------
# Visualization
# -----------------------------

def draw_graph_matplotlib(graph: nx.Graph, cover: List, node_costs: Dict) -> plt.Figure:
    # Layout: deterministic for stability across runs
    pos = nx.spring_layout(graph, seed=42)

    covered_set = set(cover)
    node_colors = ["red" if n in covered_set else "skyblue" for n in graph.nodes()]
    node_sizes = [100 + 20 * float(node_costs.get(n, 1.0)) for n in graph.nodes()]

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.8, ax=ax)

    # Label only covered nodes to reduce clutter
    labels = {n: str(n) for n in graph.nodes() if n in covered_set}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

    ax.set_axis_off()
    fig.tight_layout()
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="PLC MWVC", layout="centered")
st.title("PLC MWVC – Attenuations and MWVC")

st.write("Select your files, run the computation, and visualize the optimal solution.")

mode = st.radio(
    "File selection mode",
    ["Local files (paths)", "Upload files"],
    horizontal=True,
)

nodes_src = None
edges_src = None
zc_gamma_src = None

if mode == "Local files (paths)":
    st.subheader("File paths")
    default_nodes = "nodes.xlsx"
    default_edges = "edges.xlsx"
    default_zc = "zc_gamma_func_cenelec_b.csv"

    nodes_path = st.text_input("nodes.xlsx", value=default_nodes)
    edges_path = st.text_input("edges.xlsx", value=default_edges)
    zc_path = st.text_input("zc_gamma_func_cenelec_b.csv", value=default_zc)

    nodes_src = nodes_path
    edges_src = edges_path
    zc_gamma_src = zc_path

    col1, col2, col3 = st.columns(3)
    with col1:
        if nodes_path and os.path.exists(nodes_path):
            st.success("OK")
        else:
            st.warning("File not found")
    with col2:
        if edges_path and os.path.exists(edges_path):
            st.success("OK")
        else:
            st.warning("File not found")
    with col3:
        if zc_path and os.path.exists(zc_path):
            st.success("OK")
        else:
            st.warning("File not found")

else:
    st.subheader("Upload your files")
    nodes_up = st.file_uploader("nodes.xlsx", type=["xlsx", "xls"])  # excel only
    edges_up = st.file_uploader("edges.xlsx", type=["xlsx", "xls"])  # excel only
    zc_up = st.file_uploader("zc_gamma_func_cenelec_b.csv", type=["csv"])  # csv
    nodes_src = nodes_up
    edges_src = edges_up
    zc_gamma_src = zc_up

algorithm = st.selectbox("MWVC algorithm", ["Exact (PuLP)", "Local 2-Approx Heuristic"], index=0)
run = st.button("Run")

if run:
    if not (nodes_src and edges_src and zc_gamma_src):
        st.error("Please provide all three input files.")
    else:
        placeholder = st.empty()
        try:
            with st.spinner("Loading files…"):
                nodes_df, edges_df, zc_gamma_df = load_inputs(nodes_src, edges_src, zc_gamma_src)

            logs: List[str] = []
            with st.spinner("Building graph…"):
                G = create_graph(edges_df, nodes_df, zc_gamma_df, logs)

            if logs:
                st.info("\n".join(logs))

            st.write(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

            with st.spinner("Computing attenuations…"):
                attenuations = compute_all_attenuations(G)

            st.success("Attenuations computed.")
            att_df = pd.DataFrame(attenuations, columns=["Start Node", "End Node", "Attenuation (dB)"])
            st.write("Attenuation preview:")
            st.dataframe(att_df.head(20))

            with st.spinner(f"Solving MWVC – {algorithm}…"):
                node_costs_raw = compute_node_costs_from_attenuations(G, attenuations)
                node_costs = normalize_costs(node_costs_raw)
                if algorithm == "Exact (PuLP)":
                    cover, total_cost = solve_mwvc(G, node_costs)
                else:
                    cover = local_2_approx_mwvc(G, node_costs)
                    total_cost = float(sum(node_costs[n] for n in cover))

            st.subheader("MWVC Results")
            colA, colB = st.columns(2)
            colA.metric("Cover size", f"{len(cover)}")
            colB.metric("Total cost", f"{total_cost:.2f}")

            st.write("Nodes in the cover:")
            cover_df = pd.DataFrame({"Node": cover})
            st.dataframe(cover_df)

            with st.expander("Show graph (matplotlib)", expanded=True):
                fig = draw_graph_matplotlib(G, cover, node_costs)
                st.pyplot(fig, clear_figure=True)

            # Downloads
            att_csv = att_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download attenuations (CSV)", data=att_csv, file_name="attenuations_bfs.csv", mime="text/csv")

            costs_df = pd.DataFrame({"Node": list(node_costs.keys()), "Cost": list(node_costs.values())})
            costs_csv = costs_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download costs (CSV)", data=costs_csv, file_name="node_costs.csv", mime="text/csv")

            cover_csv = cover_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download cover (CSV)", data=cover_csv, file_name="mwvc_cover.csv", mime="text/csv")

        except Exception as exc:
            st.error(str(exc))
        finally:
            placeholder.empty() 