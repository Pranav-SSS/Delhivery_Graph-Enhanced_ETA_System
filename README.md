# Delhivery Graph-Enhanced ETA System

**IIT Guwahati CACI Summer Projects '26 — Logistics Analytics Track**

Delhivery's OSRM routing engine systematically underestimates real delivery
time (median actual/OSRM ratio = **1.86×** across 144,867 segments), causing
widespread SLA breaches. This project builds a graph-enhanced quantile
regression system that closes that gap and packages the findings into
consulting deliverables for operations leadership.

## Results

| Metric | Baseline | This model | Change |
|---|---|---|---|
| Segment MAE | 52.39 min | 37.97 min | **−28%** |
| Within-15% accuracy | 45.5% | 68.0% | **+22.5 pp** |
| Trip-level P10–P90 coverage | — | 80.7% | hits 80% target |
| FTL within-15% accuracy | 45.5% | 75.0% | oracle ceiling: 80.4% |

## Repo structure

```
delhivery-eta-project/
├── src/
│   └── delhivery_eta.py              # Full pipeline: graph build → feature
│                                      # engineering → 6-model quantile training
│                                      # → 3 consulting deliverables → evaluation
├── models/
│   └── eta_artifacts.pkl             # Trained artifacts (models, calibration
│                                      # scales, graph, corridor/strat tables)
├── dashboard/
│   ├── delhivery_dashboard.jsx       # React dashboard (Overview / Model /
│   │                                 # Network / Hubs / FTL vs Carting /
│   │                                 # Uncertainty tabs)
│   └── network_visualization.json    # Node/edge data for the network graph
│                                      # tab (also has an inline fallback
│                                      # baked into the .jsx if this file
│                                      # isn't served alongside it)
├── docs/
│   └── network_operations_strategy_memo.docx   # Leadership-facing memo
├── requirements.txt
└── .gitignore
```

## Methodology summary

- **Graph construction**: directed weighted graph over facility nodes, edge
  weight = median actual/OSRM delay factor per corridor. Betweenness
  centrality, PageRank, and edge betweenness computed via NetworkX (exact for
  ≤5k nodes, k=200 sampling approximation above that).
- **Models**: 6 LightGBM quantile regressors (P10 / P50 / P90 × FTL /
  Carting), trained on `log(actual/OSRM)` rather than raw time, which drops
  `osrm_distance`'s feature importance from 87% → 34% and surfaces real delay
  signal.
- **Calibration**: post-hoc scale factor `s = Q_α(actual/pred)` fit on a
  held-out validation split, so `P(pred·s > actual) = α` by construction.
- **Cold-start correction**: unseen corridors get a stratified fallback
  (betweenness tier × distance band × route type) multiplied by an empirical
  cold-start bias correction (FTL ×1.050, Carting ×1.144).
- **Consulting deliverables** (all derived from the same trained artifact
  dict — see `src/delhivery_eta.py`):
  - `extract_bottleneck_hubs()` — top-5 hubs by risk score
    (betweenness × delay × SLA breach rate)
  - `ftl_vs_carting_framework()` — 4-rule FTL vs. Carting decision matrix
  - `simulate_hub_upgrade()` — revenue-at-risk simulation with network
    cascade effect

## Running it

```bash
pip install -r requirements.txt
python src/delhivery_eta.py --data delivery_data.csv --save models/eta_artifacts.pkl
```

**Note on data**: `delivery_data.csv` is not included in this repo (raw
Delhivery trip/segment data — omitted for size/confidentiality). The script
expects a CSV with a `data` column containing `"training"` / `"test"` split
labels; see `FEATURES` and `build_features()` in `src/delhivery_eta.py` for
the full input schema.

To skip training and just explore the trained model:

```python
import pickle
with open("models/eta_artifacts.pkl", "rb") as f:
    arts = pickle.load(f)
# arts.keys() -> models, cal_scales, graph, corridor_stats, strat_table,
#                node_stats, edge_btw
```

Loading the pickle requires the **exact package versions** pinned in
`requirements.txt` (LightGBM/pandas/sklearn objects are version-sensitive).
If it fails to load, retrain from source instead of debugging the pickle.

## Viewing the dashboard

`dashboard/delhivery_dashboard.jsx` is a self-contained React component. To
view it:
1. Drop `delhivery_dashboard.jsx` and `network_visualization.json` into the
   same folder of any React sandbox (e.g. CodeSandbox, a local Vite/CRA app,
   or Claude/ChatGPT's artifact preview).
2. It `fetch()`es `network_visualization.json` relatively — if that fetch
   fails (e.g. file not served), it silently falls back to a small inline
   sample graph baked into the component, so the dashboard never breaks, but
   you'll see a reduced 10-node/15-edge graph instead of the full network.

## Key technical notes / gotchas

- `cutoff_factor` = minutes-until-deadline (**not** a trip-position fraction).
- `is_cutoff` = business priority flag (**not** a single-segment indicator).
- LightGBM quantile objective: `alpha=q` predicts the q-th percentile
  directly — no post-processing needed beyond calibration.
- Trip-level stage-2 stacking was tried and discarded (overfit on small
  out-of-fold trip sets); replaced with a simple mean-bias correction
  (`P90_SLA = P90 + 5.3 min/segment`) for capacity-planning use cases.

## Built By
Pranav SSS

## License

MIT — see `LICENSE`.
