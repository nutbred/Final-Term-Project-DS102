# AromaLatent ML Track

This package implements the Machine Learning / Data Science branch of AromaLatent using structured Fragrantica descriptors only.

## Run Order

```powershell
python -m ml_track.features
python -m ml_track.dupe_detector --top-k 10
python -m ml_track.cluster_space --clusters 14
python -m ml_track.note_graph
```

For a quick smoke test:

```powershell
python -m ml_track.features
python -m ml_track.dupe_detector --top-k 5 --limit 100
python -m ml_track.cluster_space --clusters 10 --sample-size 1000
python -m ml_track.note_graph --min-note-count 80 --min-pair-count 20
```

## Outputs

- `output/ml/perfume_features.npz`
- `output/ml/perfume_index.csv`
- `output/ml/feature_names.json`
- `output/ml/dupe_candidates.csv`
- `output/ml/clusters.csv`
- `output/ml/latent_space_2d.html`
- `output/ml/latent_space_3d.html`
- `output/ml/cluster_report.md`
- `output/ml/note_centrality.csv`
- `output/ml/note_communities.csv`
- `output/ml/note_graph.graphml`
- `output/ml/note_graph_report.md`

`umap-learn` and `hdbscan` are optional. If they are not installed, `cluster_space.py` falls back to SVD coordinates and KMeans clustering.

