# PLC MWVC – Power Line Communications Graph Attenuations and Minimum Weighted Vertex Cover

This repository contains code and experiments to:
- compute end-to-end attenuations on a graph representing a PLC (Power Line Communications) network using ABCD matrices, and
- solve the Minimum Weighted Vertex Cover (MWVC) based on costs derived from those attenuations.

A lightweight Streamlit GUI is included to run the full pipeline on your own data files and visualize the optimal cover.

---

## Quick start (GUI)

Requirements:
- Python 3.9+
- Packages: `streamlit`, `pandas`, `numpy`, `networkx`, `pulp`, `openpyxl`, `matplotlib`

Install dependencies:
```bash
cd "/Users/pellandini/Documents/Master/Master/Code/MA_Vertex_Cover"
python -m pip install --upgrade pip
pip install -r requirements.txt  # optional if you keep it updated
# or install explicitly:
pip install streamlit pandas numpy networkx pulp openpyxl matplotlib
```

Launch the GUI:
```bash
streamlit run "Code/GUI/PLC_MWVC_Streamlit.py"
```

What you can do with the GUI:
- Select your three inputs and run the full computation.
- See a live progress bar while BFS-based attenuation aggregation runs.
- Inspect a preview of attenuations.
- Solve MWVC (PuLP/ CBC) with node costs derived from attenuations.
- Visualize the graph with covered nodes highlighted (red) and other nodes in blue; node size scales with normalized cost.
- Download CSVs: attenuations, node costs, and cover.

---

## Data format

- Nodes file (`nodes.xlsx`): must contain column `ID_NODE`.
- Edges file (`edges.xlsx`): must contain columns `ID_FROM_NODE`, `ID_TO_NODE`. Optional columns used if present:
  - `LENGTH`: segment length
  - `DIAMETER` (or `KEY`): used to look up line parameters
- ZC/Gamma file (`zc_gamma_func_cenelec_b.csv`): CSV with columns:
  - `KEY` (matching `DIAMETER` in edges),
  - `rGAMMA`, `iGAMMA` (real/imag of propagation constant),
  - `rZC`, `iZC` (real/imag of characteristic impedance).

If some parameters are missing, the GUI treats the corresponding edges as identity (no additional attenuation) to remain robust.

---

## How it works (overview)

1) ABCD matrix per edge: for a segment of length L with propagation constant γ and characteristic impedance Zc, the ABCD matrix is built with cosh/sinh terms. For a path, matrices are multiplied to get the total transfer.

2) Attenuation per pair: a BFS is run from each node, accumulating ABCD along the tree edges; for each reached node, we compute a final attenuation proxy from the accumulated matrix.

3) Node costs: for every pair (u, v) we add |attenuation(u, v)| to both u and v, then normalize costs to a 1–100 range.

4) MWVC: we solve
   minimize Σ c(v) x_v subject to x_u + x_v ≥ 1 for all edges (u, v), x_v ∈ {0, 1}.

5) Visualization: the resulting cover is highlighted on the graph; covered nodes in red, others in blue.

---

## Repository structure

- `Code/GUI/`
  - `PLC_MWVC_Streamlit.py`: Streamlit app for selecting files, running computations, visualization, and CSV export.
  - `nodes.xlsx`, `edges.xlsx`, `zc_gamma_func_cenelec_b.csv`: sample inputs.

- `Code/Tree MWVC/`
  - `PLC_MWVC.py`: CLI script to compute attenuations and solve MWVC (no GUI).
  - `MWVC_SIG.py`: baseline/signature implementation for MWVC on tree graphs.
  - `nodes.xlsx`, `edges.xlsx`, `zc_gamma_func_cenelec_b.csv`: sample inputs.
  - `attenuations_bfs.csv`: example output (large file).

- `Experiments/`
  - `Quantum/`: experimental quantum-related scripts.

- `Organisation/`
  - planning and organization assets.

- `requirements.txt`: optional pinned dependencies for convenience.

---

## CLI usage (alternative to GUI)

You can also run the pipeline from the command line:
```bash
python "Code/Tree MWVC/PLC_MWVC.py" \
  --nodes "Code/Tree MWVC/nodes.xlsx" \
  --edges "Code/Tree MWVC/edges.xlsx" \
  --zc-gamma "Code/Tree MWVC/zc_gamma_func_cenelec_b.csv" \
  --out-atten "attenuations_bfs.csv" \
  --n-jobs -1
```

---

## Troubleshooting

- Streamlit warning "run via streamlit run": always start the GUI with `streamlit run Code/GUI/PLC_MWVC_Streamlit.py`.
- Missing `_tkinter`: not used here; the GUI is Streamlit-based.
- CBC solver not found / PuLP errors: ensure `pulp` is installed; the default CBC bundled with PuLP is used by default.
- Excel read errors: install `openpyxl`.

---

## License

This project is for academic use. If you plan to use it in production or redistribute, please add an explicit license file and credit the authors. 